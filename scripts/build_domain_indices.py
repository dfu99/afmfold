import argparse
from pathlib import Path

import mdtraj as md
import numpy as np


def chain_id_for(chain):
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
                "chain_id": chain_id_for(chain),
                "chain_index": chain.index,
                "res_seq": res.resSeq,
                "res_name": res.name,
            }
        )
    return ca_info


def select_chain_indices(ca_info, chain_id):
    return np.array([i for i, info in enumerate(ca_info) if info["chain_id"] == chain_id], dtype=int)


def auto_select_chains(ca_info):
    counts = {}
    for info in ca_info:
        counts[info["chain_id"]] = counts.get(info["chain_id"], 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) < 2:
        raise ValueError("Need at least two chains to build chain_a and chain_b.")
    return ranked[0][0], ranked[1][0], counts


def main():
    parser = argparse.ArgumentParser(
        description="Build chain_a.npy and chain_b.npy from CA ordering of a PDB/CIF."
    )
    parser.add_argument("--pdb", required=True, help="Path to the PDB/CIF file.")
    parser.add_argument("--out-dir", default="storage/domain/integrin", help="Output directory.")
    parser.add_argument("--chain-a-id", help="Chain ID to use for chain_a.npy.")
    parser.add_argument("--chain-b-id", help="Chain ID to use for chain_b.npy.")
    args = parser.parse_args()

    traj = md.load(args.pdb)
    ca_info = collect_ca_info(traj)

    if args.chain_a_id and args.chain_b_id:
        chain_a_id = str(args.chain_a_id)
        chain_b_id = str(args.chain_b_id)
        counts = {}
        for info in ca_info:
            counts[info["chain_id"]] = counts.get(info["chain_id"], 0) + 1
    else:
        chain_a_id, chain_b_id, counts = auto_select_chains(ca_info)

    chain_a = select_chain_indices(ca_info, chain_a_id)
    chain_b = select_chain_indices(ca_info, chain_b_id)

    if chain_a.size == 0 or chain_b.size == 0:
        raise ValueError(
            f"Empty selection: chain_a={chain_a_id} (n={chain_a.size}), "
            f"chain_b={chain_b_id} (n={chain_b.size})."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "chain_a.npy", chain_a)
    np.save(out_dir / "chain_b.npy", chain_b)

    print("CA chain counts:", counts)
    print(f"chain_a: {chain_a_id} -> {len(chain_a)} CA indices")
    print(f"chain_b: {chain_b_id} -> {len(chain_b)} CA indices")
    print(f"Saved to: {out_dir / 'chain_a.npy'}")
    print(f"Saved to: {out_dir / 'chain_b.npy'}")


if __name__ == "__main__":
    main()
