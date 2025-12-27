import argparse
from pathlib import Path

import mdtraj as md
import numpy as np


def load_domain_indices(domain_path):
    domain_path = Path(domain_path)
    if not domain_path.exists():
        raise FileNotFoundError(f"Missing domain file: {domain_path}")
    arr = np.load(domain_path)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array in {domain_path}, got {arr.shape}")
    return arr.astype(int)


def summarize_domain(name, indices, ca_info):
    n_ca = len(ca_info)
    summary = {
        "name": name,
        "n_indices": len(indices),
        "min_index": int(indices.min()) if len(indices) else None,
        "max_index": int(indices.max()) if len(indices) else None,
        "out_of_range": np.any(indices < 0) or np.any(indices >= n_ca),
        "unique": len(np.unique(indices)) == len(indices),
    }

    chain_ids = [ca_info[i]["chain_id"] for i in indices if 0 <= i < n_ca]
    summary["chain_id_counts"] = {cid: chain_ids.count(cid) for cid in sorted(set(chain_ids))}

    sample = []
    for i in indices[:10]:
        if 0 <= i < n_ca:
            info = ca_info[i]
            sample.append(f"{info['chain_id']}:{info['res_seq']}{info['res_name']}")
    summary["sample"] = sample
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Check whether chain_a.npy/chain_b.npy match CA ordering in a PDB."
    )
    parser.add_argument("--pdb", required=True, help="Path to the reference PDB/CIF.")
    parser.add_argument(
        "--chain-a",
        default="storage/domain/integrin/chain_a.npy",
        help="Path to chain_a.npy (CA indices).",
    )
    parser.add_argument(
        "--chain-b",
        default="storage/domain/integrin/chain_b.npy",
        help="Path to chain_b.npy (CA indices).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with nonzero code on any mismatch.",
    )
    args = parser.parse_args()

    traj = md.load(args.pdb)
    ca_indices = [atom.index for atom in traj.topology.atoms if atom.name == "CA"]
    ca_traj = traj.atom_slice(ca_indices)

    ca_info = []
    for atom in ca_traj.topology.atoms:
        res = atom.residue
        chain = res.chain
        chain_id = getattr(chain, "id", None)
        if chain_id is None:
            chain_id = getattr(chain, "name", None)
        if chain_id is None:
            chain_id = str(chain.index)
        ca_info.append(
            {
                "chain_id": chain_id,
                "chain_index": chain.index,
                "res_seq": res.resSeq,
                "res_name": res.name,
            }
        )

    chain_a = load_domain_indices(args.chain_a)
    chain_b = load_domain_indices(args.chain_b)

    summaries = [
        summarize_domain("chain_a", chain_a, ca_info),
        summarize_domain("chain_b", chain_b, ca_info),
    ]

    has_error = False
    print(f"Total CA atoms: {len(ca_info)}")
    for s in summaries:
        print(f"\n[{s['name']}]")
        print(f"  n_indices: {s['n_indices']}")
        print(f"  min_index: {s['min_index']}")
        print(f"  max_index: {s['max_index']}")
        print(f"  unique: {s['unique']}")
        print(f"  out_of_range: {s['out_of_range']}")
        print(f"  chain_id_counts: {s['chain_id_counts']}")
        print(f"  sample (first 10): {s['sample']}")
        if s["out_of_range"] or not s["unique"]:
            has_error = True

    if args.strict and has_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
