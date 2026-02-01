#!/usr/bin/env python3
import argparse
from pathlib import Path

import mdtraj as md
import numpy as np

DEFAULT_HEAD_RANGES = [
    ("A", 1, 435),
    ("B", 113, 352),
]
DEFAULT_TAIL_RANGES = [
    ("A", 742, 962),
    ("B", 607, 692),
]


def get_chain_id(chain):
    chain_id = getattr(chain, "chain_id", None)
    if chain_id is None:
        chain_id = getattr(chain, "id", None)
    if chain_id is None:
        chain_id = getattr(chain, "name", None)
    if chain_id is None:
        chain_id = str(chain.index)
    return str(chain_id)


def collect_ca_info(traj):
    ca_indices = [atom.index for atom in traj.topology.atoms if atom.name == "CA"]
    ca_traj = traj.atom_slice(ca_indices)
    ca_info = []
    for atom in ca_traj.topology.atoms:
        res = atom.residue
        chain = res.chain
        ca_info.append(
            {
                "chain_id": get_chain_id(chain),
                "res_seq": res.resSeq,
            }
        )
    return ca_info


def select_by_ranges(ca_info, chain_id, ranges):
    out = []
    for i, info in enumerate(ca_info):
        if info["chain_id"] != chain_id:
            continue
        resseq = info["res_seq"]
        for start, end in ranges:
            if start <= resseq <= end:
                out.append(i)
                break
    return np.array(sorted(out), dtype=int)


def select_chain(ca_info, chain_id):
    return np.array([i for i, info in enumerate(ca_info) if info["chain_id"] == chain_id], dtype=int)


def main():
    parser = argparse.ArgumentParser(
        description="Build AVB3 domain indices from head/tail residue ranges."
    )
    parser.add_argument(
        "--pdb",
        default="storage/AVB3_clean_nowater.pdb",
        help="Reference PDB path.",
    )
    parser.add_argument(
        "--out-dir",
        default="storage/domain/avb3",
        help="Output directory for .npy files.",
    )
    args = parser.parse_args()

    traj = md.load(args.pdb)
    ca_info = collect_ca_info(traj)

    head_by_chain = {}
    tail_by_chain = {}
    for chain_id, start, end in DEFAULT_HEAD_RANGES:
        head_by_chain.setdefault(chain_id, []).append((start, end))
    for chain_id, start, end in DEFAULT_TAIL_RANGES:
        tail_by_chain.setdefault(chain_id, []).append((start, end))

    for chain_id in ["A", "B"]:
        if chain_id not in head_by_chain or chain_id not in tail_by_chain:
            raise ValueError(f"Missing head/tail ranges for chain {chain_id}")

    chain_a = select_chain(ca_info, "A")
    chain_b = select_chain(ca_info, "B")

    alpha_head = select_by_ranges(ca_info, "A", head_by_chain["A"])
    alpha_tail = select_by_ranges(ca_info, "A", tail_by_chain["A"])
    beta_head = select_by_ranges(ca_info, "B", head_by_chain["B"])
    beta_tail = select_by_ranges(ca_info, "B", tail_by_chain["B"])

    alpha_mid = np.setdiff1d(chain_a, np.union1d(alpha_head, alpha_tail), assume_unique=True)
    beta_mid = np.setdiff1d(chain_b, np.union1d(beta_head, beta_tail), assume_unique=True)
    alpha_coil = np.array([], dtype=int)
    beta_coil = np.array([], dtype=int)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "chain_a.npy", chain_a)
    np.save(out_dir / "chain_b.npy", chain_b)

    np.save(out_dir / "alpha_head_thigh.npy", alpha_head)
    np.save(out_dir / "alpha_calf.npy", alpha_mid)
    np.save(out_dir / "alpha_coil.npy", alpha_coil)

    np.save(out_dir / "beta_head_hybrid_egf1.npy", beta_head)
    np.save(out_dir / "beta_tail_egf2_3_4_btail.npy", beta_mid)
    np.save(out_dir / "beta_coil.npy", beta_coil)

    print(f"Saved AVB3 domain indices to {out_dir}")
    print(f"chain_a: {len(chain_a)} CA")
    print(f"chain_b: {len(chain_b)} CA")
    print(f"alpha_head_thigh: {len(alpha_head)} CA")
    print(f"alpha_calf: {len(alpha_mid)} CA")
    print(f"alpha_coil: {len(alpha_coil)} CA")
    print(f"beta_head_hybrid_egf1: {len(beta_head)} CA")
    print(f"beta_tail_egf2_3_4_btail: {len(beta_mid)} CA")
    print(f"beta_coil: {len(beta_coil)} CA")


if __name__ == "__main__":
    main()
