from pathlib import Path
import numpy as np

# Refer to storage/domain
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DOMAIN_DIR = BASE_DIR / "storage" / "domain"

AK_DOMAINS = {
    "ATPbd": np.load(DOMAIN_DIR / "ak/atpbd.npy"),
    "Core": np.load(DOMAIN_DIR / "ak/core.npy"),
    "AMPbd": np.load(DOMAIN_DIR / "ak/ampbd.npy"),
}

FLHAC_DOMAINS = {
    r"A$_C$D$_1$": np.load(DOMAIN_DIR / "flhac/acd1.npy"),
    r"A$_C$D$_2$": np.load(DOMAIN_DIR / "flhac/acd2.npy"),
    r"A$_C$D$_3$": np.load(DOMAIN_DIR / "flhac/acd3.npy"),
    r"A$_C$D$_4$": np.load(DOMAIN_DIR / "flhac/acd4.npy"),
}

def get_domain_pair_names(protein_name="4ake"):
    if protein_name.lower() in ["3a5i", "flhac"]:
        domain_pair_names = [
            (r"A$_C$D$_2$", r"A$_C$D$_3$"),
            (r"A$_C$D$_2$", r"A$_C$D$_4$"),
            (r"A$_C$D$_4$", r"A$_C$D$_1$"),
        ]
    
    elif protein_name.lower() in ["1ake", "4ake", "ak"]:
        domain_pair_names = [
            ("ATPbd", "Core"),
            ("Core", "AMPbd"),
            ("AMPbd", "ATPbd"),
        ]
    
    else:
        raise NotImplementedError(f"Invalid Protein Name: {protein_name}")
    
    return domain_pair_names

def get_domain_pairs(protein_name="4ake"):
    domain_pair_names = get_domain_pair_names(protein_name)
    if protein_name.lower() in ["3a5i", "flhac"]:
        domain_pairs = [
            (FLHAC_DOMAINS[name1], FLHAC_DOMAINS[name2])
            for name1, name2 in domain_pair_names
        ]
    
    elif protein_name.lower() in ["1ake", "4ake", "ak"]:
        domain_pairs = [
            (AK_DOMAINS[name1], AK_DOMAINS[name2])
            for name1, name2 in domain_pair_names
        ]
    
    else:
        raise NotImplementedError(f"Invalid Protein Name: {protein_name}")
    
    return domain_pairs

def compute_domain_distance(traj, domain1, domain2):
    """
    Function to project a trajectory [B, N, 3] into domain distance [B, 1]

    Args:
        traj (mdtraj.Trajectory): trajectory
        domain1 (list of int): residue indices forming domain 1
        domain2 (list of int): residue indices forming domain 2

    Returns:
        np.ndarray: domain distance [B, 1]
    """
    # Extract only Cα atoms from traj
    ca_indices = [atom.index for atom in traj.topology.atoms if atom.name == 'CA']
    traj = traj.atom_slice(ca_indices)
    
    # Get the number of residues
    num_atoms = traj.xyz.shape[1]  # traj.xyz.shape = [B, N, 3]

    # Input validation: check if indices are within range
    if max(domain1) >= num_atoms or max(domain2) >= num_atoms:
        raise ValueError("Indices of domain1 or domain2 exceed the number of atoms in traj.")

    # Extract coordinates of each domain
    coords1 = traj.xyz[:, domain1, :]  # shape: [B, len(domain1), 3]
    coords2 = traj.xyz[:, domain2, :]  # shape: [B, len(domain2), 3]

    # Compute the mean coordinates (centroid) of each domain
    centroid1 = np.mean(coords1, axis=1)  # shape: [B, 3]
    centroid2 = np.mean(coords2, axis=1)  # shape: [B, 3]

    # Compute the Euclidean distance between centroids for each frame
    distances = np.linalg.norm(centroid1 - centroid2, axis=1, keepdims=True)  # shape: [B, 1]

    return distances
