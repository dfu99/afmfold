import numpy as np
import mdtraj as md
import math
from tqdm import tqdm
from pathlib import Path
import argparse

from afmfold.domain import get_domain_pairs, compute_domain_distance
from afmfold.inference import conditional_generation

# Refer to storage/domain
BASE_DIR = Path(__file__).resolve().parent.parent

def make_candidates(
    native_pdb, 
    name, 
    max_deviation=50.0,  # in Å
    grid_size=5.0,  # in Å
    out_dir=BASE_DIR / "inference" / "training",
    json_file=BASE_DIR / "storage" / "3a5i.json",
    r_neighbors=2,
    min_trial=20,
    max_trial=1,
    delete_failed_cifs=True,
    ):
    # Initializations.
    native_traj = md.load(native_pdb)
    domain_pairs = get_domain_pairs(name)
    
    R = len(native_traj)
    D = len(domain_pairs)
    
    # Compute domain distance of native structure.
    native_domain_distances = np.zeros((R, D))
    for i, (d1, d2) in enumerate(domain_pairs):
        d = compute_domain_distance(native_traj, d1, d2)
        native_domain_distances[:,i] = d.ravel()
    
    native_domain_distances *= 10.0  # from nm to Å
    
    # Make grid
    min_dd = np.min(native_domain_distances, axis=0) - max_deviation
    max_dd = np.max(native_domain_distances, axis=0) + max_deviation
    num_half_grids = [math.ceil((max_dd[i] - min_dd[i]) / 2 / grid_size) for i in range(len(domain_pairs))]
    centers = [0.5 * (max_dd[i] + min_dd[i]) for i in range(len(domain_pairs))]
    _grids_per_axis = [centers[i] + grid_size * np.arange(-num_half_grids[i], num_half_grids[i]+1) for i in range(len(domain_pairs))]
    grids_per_axis = [g[g > 0] for g in _grids_per_axis]
    mesh_grids = np.meshgrid(*grids_per_axis)
    grids = np.concatenate([mg.reshape((-1,1)) for mg in mesh_grids], axis=-1)
    assert grids.shape[1] == D and grids.ndim == 2
    
    # Sort by distance.
    mean_native_domain_distances = np.mean(native_domain_distances, axis=0)
    delta = np.sum((grids - mean_native_domain_distances[None,:])**2, axis=-1)
    decreasing_order = np.argsort(delta)
    grids = grids[decreasing_order,:]
    assert grids.shape[1] == D and grids.ndim == 2
    
    # Start generating.
    success_dict = {}
    for i in tqdm(range(len(grids)), total=len(grids)):
        # End if neighboring trial has been all failed.
        deltas = np.sum(np.abs(grids - grids[i][None,:]), axis=-1)
        neighbor_ind = np.nonzero(deltas < grid_size * r_neighbors)[0]
        
        if i < min_trial or any(j in success_dict and success_dict[j] for j in neighbor_ind):
            pass
        else:
            continue
        
        successful_cif_list, failed_cif_list = conditional_generation(
            target_domain_distance=grids[i],
            domain_pairs=domain_pairs,
            out_dir=str(out_dir),
            name=name,
            json_file=str(json_file),
            input_dict={},
            max_trial=max_trial,
            )
        
        if len(successful_cif_list) > 0:
            success_dict[i] = True
        else:
            success_dict[i] = False
        
        # Delete unsuccessful results.
        if delete_failed_cifs:
            import shutil
            failed_dirs = [Path(cif).resolve().parent.parent for cif in failed_cif_list]
            for d in failed_dirs:
                shutil.rmtree(d)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse arguments for training AFM image generator.")
    parser.add_argument('--native-pdb', type=str, required=True, help='Path to the reference PDB file.')
    parser.add_argument('--name', type=str, required=True, help='Name of the protein')
    parser.add_argument('--json-file', type=str, required=True, help='Json path')
    parser.add_argument('--out-dir', type=str, required=True, help='Output dir')
    parser.add_argument('--max-deviation', type=float, default=5.0, help='Maximum deviation in Angstrom')
    parser.add_argument('--grid-size', type=float, default=5.0, help='Grid size in Angstrom')
    parser.add_argument('--max-trial', type=int, default=1, help='Maximum trials per grid point')
    args = parser.parse_args()
    
    print(
        f"\nGenerating candidates with parameters:\n"
        f"  native_pdb   : {args.native_pdb}\n"
        f"  name         : {args.name}\n"
        f"  json_file    : {args.json_file}\n"
        f"  out_dir      : {args.out_dir}\n"
        f"  max_deviation: {args.max_deviation}\n"
        f"  grid_size    : {args.grid_size}\n"
        f"  max_trial    : {args.max_trial}\n"
    )
    make_candidates(
        args.native_pdb,
        args.name,
        out_dir=args.out_dir,
        json_file=args.json_file,
        max_deviation=args.max_deviation,
        grid_size=args.grid_size,
        max_trial=args.max_trial,
    )