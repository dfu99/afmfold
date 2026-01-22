#!/usr/bin/env python3
import argparse
import glob
import os
import sys


STANDARD_AA = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Verify that stripped PDBs contain identical chains, residue numbers, "
            "and sequences (no missing/excess residues or ligands)."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="storage/avb3_refs/stripped_pdb",
        help="Directory containing stripped PDB files.",
    )
    parser.add_argument(
        "--reference",
        default="",
        help="Optional reference PDB file; defaults to first in sorted list.",
    )
    parser.add_argument(
        "--allow-hetatm",
        action="store_true",
        help="Allow HETATM records without failing.",
    )
    return parser.parse_args()


def parse_pdb(path):
    chains = {}
    hetatm_count = 0
    nonstandard = set()
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for line in fh:
            if line.startswith("HETATM"):
                hetatm_count += 1
                continue
            if not line.startswith("ATOM"):
                continue
            if len(line) < 54:
                continue
            resname = line[17:20].strip()
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            if resname not in STANDARD_AA:
                nonstandard.add(resname)
            key = (int(resseq), icode, resname)
            chains.setdefault(chain_id, [])
            if not chains[chain_id] or chains[chain_id][-1] != key:
                chains[chain_id].append(key)
    return chains, hetatm_count, sorted(nonstandard)


def chain_signature(chain):
    residues = [(rnum, icode) for rnum, icode, _ in chain]
    seq = "".join(STANDARD_AA.get(res, "X") for _, _, res in chain)
    return residues, seq


def compare(reference, target, ref_name, tgt_name, issues):
    ref_chains = set(reference.keys())
    tgt_chains = set(target.keys())
    if ref_chains != tgt_chains:
        issues.append(
            f"{tgt_name}: chain IDs differ (ref={sorted(ref_chains)} "
            f"target={sorted(tgt_chains)})"
        )
        return
    for chain_id in sorted(ref_chains):
        ref_res, ref_seq = chain_signature(reference[chain_id])
        tgt_res, tgt_seq = chain_signature(target[chain_id])
        if ref_res != tgt_res:
            issues.append(
                f"{tgt_name}: chain {chain_id} residue numbers differ "
                f"(ref={len(ref_res)} target={len(tgt_res)})"
            )
        if ref_seq != tgt_seq:
            issues.append(
                f"{tgt_name}: chain {chain_id} sequence differs "
                f"(ref_len={len(ref_seq)} target_len={len(tgt_seq)})"
            )


def main():
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.pdb")))
    if not paths:
        raise SystemExit(f"No PDB files found in {args.input_dir}")

    if args.reference:
        ref_path = args.reference
    else:
        ref_path = paths[0]

    ref_chains, ref_hetatm, ref_nonstandard = parse_pdb(ref_path)
    issues = []
    if ref_hetatm and not args.allow_hetatm:
        issues.append(f"{os.path.basename(ref_path)}: HETATM records present")
    if ref_nonstandard:
        issues.append(
            f"{os.path.basename(ref_path)}: nonstandard residues {ref_nonstandard}"
        )

    for path in paths:
        chains, hetatm_count, nonstandard = parse_pdb(path)
        if hetatm_count and not args.allow_hetatm:
            issues.append(f"{os.path.basename(path)}: HETATM records present")
        if nonstandard:
            issues.append(f"{os.path.basename(path)}: nonstandard residues {nonstandard}")
        compare(ref_chains, chains, ref_path, os.path.basename(path), issues)

    if issues:
        print("Mismatch report:")
        for item in issues:
            print(f"- {item}")
        raise SystemExit(1)

    print(f"All PDBs match reference: {os.path.basename(ref_path)}")


if __name__ == "__main__":
    main()
