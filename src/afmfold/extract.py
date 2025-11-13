from collections import defaultdict
import mdtraj as md
import numpy as np
from typing import Dict, Tuple, List, Union
from pathlib import Path

from afmfold.utils import load_json, gather_traj, save_json

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BOND_JSON = BASE_DIR / "storage" / "bond.json"

def get_bond_pairs_and_types(traj: md.Trajectory, **kwargs) -> Tuple[np.ndarray, List[str]]:
    # Obtain bond information from the topology of the mdtraj trajectory
    bonds = list(traj.topology.bonds)
    pairs = np.array([[b[0].index, b[1].index] for b in bonds], dtype=int)
    
    # Map atomic type names to element symbols
    bond_types = []
    for b in bonds:
        # In mdtraj, atom.element.symbol is typically used
        elem1 = b[0].element.symbol if b[0].element is not None else b[0].name.lower()
        elem2 = b[1].element.symbol if b[1].element is not None else b[1].name.lower()
        
        # Sort to make the key order-independent
        bond_type = "-".join(sorted([elem1, elem2]))
        bond_types.append(bond_type)

    return pairs, bond_types


def build_bond_type_dict(json_path: Union[str, Path] = BOND_JSON, **kwargs) -> Dict[str, Dict[str, float]]:
    return load_json(json_path)


def compute_bond_distances(traj: md.Trajectory, pairs: np.ndarray, batch_size: int = 10000) -> np.ndarray:
    n_bonds = pairs.shape[0]
    n_frames = traj.n_frames

    # Preallocate the output array (float32 is sufficient)
    distances = np.empty((n_frames, n_bonds), dtype=np.float32)

    # Process pairs in batches
    for start in range(0, n_bonds, batch_size):
        end = min(start + batch_size, n_bonds)
        batch_pairs = pairs[start:end]

        # Compute distances using mdtraj (unit: nm)
        batch_dist = md.compute_distances(traj, batch_pairs)

        # Store the results
        distances[:, start:end] = batch_dist

    return distances


def evaluate_bond_validity(
    distances: np.ndarray,
    bond_types: List[str],
    ref_dict: Dict[str, Dict[str, float]],
    **kwargs
) -> np.ndarray:
    n_frames, n_bonds = distances.shape
    assert len(bond_types) == n_bonds
    
    probs = np.zeros_like(distances, dtype=np.float32)

    # Constant term (normal distribution normalization factor)
    sqrt_2pi = np.sqrt(2.0 * np.pi)

    # Compute probabilities for each bond type
    for i, btype in enumerate(bond_types):
        if btype not in ref_dict:
            # Unknown bond type → mark uniformly as NaN
            probs[:, i] = np.nan
            continue

        mu = ref_dict[btype]["mean"]
        sigma = ref_dict[btype]["std"]

        # Compute the normal distribution PDF
        x = distances[:, i]
        z = (x - mu) / sigma
        pdf = np.exp(-0.5 * z**2) / (sigma * sqrt_2pi)

        # Scale the PDF to make the maximum value 1 (relative score)
        pdf /= pdf.max() if pdf.max() > 0 else 1.0
        probs[:, i] = pdf.astype(np.float32)

    return probs


def classify_bond_status(
    probs: np.ndarray,
    threshold_favored: float = 0.61,
    threshold_outlier: float = 0.14,
    **kwargs
) -> np.ndarray:
    """
    0 = favored, 1 = allowed, 2 = outlier
    """
    # Temporarily replace NaN values with -1
    probs_safe = np.nan_to_num(probs, nan=-1.0)

    # Initialize (-1: undefined)
    labels = np.full_like(probs_safe, fill_value=-1, dtype=np.int8)

    # favored: p > threshold_favored
    labels[probs_safe > threshold_favored] = 0

    # allowed: threshold_outlier < p <= threshold_favored
    mask_allowed = (probs_safe <= threshold_favored) & (probs_safe > threshold_outlier)
    labels[mask_allowed] = 1

    # outlier: p <= threshold_outlier
    labels[probs_safe <= threshold_outlier] = 2

    return labels


def summarize_bond_statistics(labels: np.ndarray, bond_types: List[str], **kwargs) -> Dict[str, Dict[str, float]]:
    # Initial processing
    if labels.ndim == 1:
        labels = labels.reshape((1, -1))
    n_frames, n_bonds = labels.shape
    assert len(bond_types) == n_bonds, "The length of bond_types does not match the number of labels."

    # Exclude undefined values such as NaN or -1
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    valid_types = np.array(bond_types * n_frames)[valid_mask.flatten()]

    # Aggregate by bond type
    summary: Dict[str, Dict[str, float]] = {}
    unique_types = np.unique(valid_types)

    for btype in unique_types:
        mask = valid_types == btype
        sub_labels = valid_labels[mask]
        total = len(sub_labels)

        if total == 0:
            continue

        favored = np.sum(sub_labels == 0)
        allowed = np.sum(sub_labels == 1)
        outlier = np.sum(sub_labels == 2)

        summary[btype] = {
            "favored_ratio": favored / total,
            "allowed_ratio": (favored + allowed) / total,
            "outlier_ratio": outlier / total,
        }

    # Overall statistics
    total = len(valid_labels)
    favored_total = np.sum(valid_labels == 0)
    allowed_total = np.sum(valid_labels == 1)
    outlier_total = np.sum(valid_labels == 2)

    summary["overall"] = {
        "favored_ratio": favored_total / total if total > 0 else np.nan,
        "allowed_ratio": (favored_total + allowed_total) / total if total > 0 else np.nan,
        "outlier_ratio": outlier_total / total if total > 0 else np.nan,
    }

    return summary


def evaluate_bond_length(
    traj,
    **kwargs,
    ):
    """
    Parameters
    ----------
    traj : mdtraj.Trajectory
        Path to the trajectory file.
    top_path : str
        Path to the topology file.

    Returns
    -------
    summary : dict
        Validity evaluation summary for each bond type.
    """
    n_frames = len(traj)
    
    # Load bond statistical reference data
    ref_dict = build_bond_type_dict(**kwargs)
    
    # Obtain bond pairs
    pairs, bond_types = get_bond_pairs_and_types(traj, **kwargs)
    
    # Compute bond distances
    distances = compute_bond_distances(traj, pairs, **kwargs)
    
    # Compute statistical probabilities for each bond
    prob = evaluate_bond_validity(distances, bond_types, ref_dict)
    
    # Evaluate bond status
    labels = classify_bond_status(prob)
    
    # Process each frame
    summeries = {}
    for iframe in range(n_frames):
        # Compute per-frame summary
        summery = summarize_bond_statistics(labels[iframe], bond_types, **kwargs)
        
        # Aggregate per-bond results
        for bond_type, subsummery in summery.items():
            if bond_type not in summeries:
                summeries[bond_type] = {}
            
            # Aggregate by evaluation category
            for ratio_type, ratio in subsummery.items():
                if ratio_type not in summeries[bond_type]:
                    summeries[bond_type][ratio_type] = []
                
                summeries[bond_type][ratio_type].append(ratio)
    
    summeries_np = {
        bond_type: {
            ratio_type: np.array(ratios)
            for ratio_type, ratios in subsummery.items()
        }
        for bond_type, subsummery in summeries.items()
    }
    return summeries_np


THRESHOLDS = {
    "MolProbity Score": {"acceptable": "lower", "value": 4.0},
    "Clash Score": {"acceptable": "lower", "value": 70.0},
    "Rama Favored %": {"acceptable": "upper", "value": 90.0},
    "Rotamer Outlier %": {"acceptable": "lower", "value": 50.0},
}

def extract_acceptables(molprobity_results, bond_summeries, thresholds=THRESHOLDS):
    acceptable_mask = None
    for name, score in molprobity_results.items():
        if acceptable_mask is None:
            acceptable_mask = np.ones_like(score).astype(bool)
            
        if name in thresholds:
            if thresholds[name]["acceptable"] == "lower":
                current_acceptable = score < thresholds[name]["value"]
            elif thresholds[name]["acceptable"] == "upper":
                current_acceptable = score > thresholds[name]["value"]
            else:
                raise NotImplementedError()
        acceptable_mask = np.logical_and(acceptable_mask, current_acceptable)

    for name, values in bond_summeries.items():
        if name in thresholds:
            if thresholds[name]["acceptable"] == "lower":
                current_acceptable = values < thresholds[name]["value"]
            elif thresholds[name]["acceptable"] == "upper":
                current_acceptable = values > thresholds[name]["value"]
            else:
                raise NotImplementedError()
        acceptable_mask = np.logical_and(acceptable_mask, current_acceptable)
    return acceptable_mask

def select_frames(molprobity_json, thresholds=THRESHOLDS, traj=None, is_tqdm=True, return_metrics=True):
    # Load files
    mp_scores = load_json(molprobity_json)
    
    # Gather results
    molprobity_results = defaultdict(list)
    cif_files = []
    for cif_file, metrics in mp_scores.items():
        cif_files.append(cif_file)
        for name, value in metrics.items():
            molprobity_results[name].append(value)
    
    molprobity_results = {k: np.array(v) for k, v in molprobity_results.items()}
    if traj is None:
        traj = gather_traj(cif_files, is_tqdm=is_tqdm)
    else:
        assert len(traj) == len(cif_files)
    
    # Evaluate bond length
    bond_summeries = evaluate_bond_length(traj)["overall"]
    
    # Mask acceptable frames
    acceptable = extract_acceptables(molprobity_results, bond_summeries, thresholds=thresholds)
    
    if return_metrics:
        return traj, acceptable, {"molprobity_results": molprobity_results, "bond_summeries": bond_summeries}
    else:
        return traj, acceptable