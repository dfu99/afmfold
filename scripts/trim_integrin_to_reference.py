#!/usr/bin/env python3
import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate trim ranges against a reference PDB and write trimmed PDBs "
            "with identical chains/residue sets."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="storage/avb3_refs/stripped_pdb",
        help="Directory containing stripped PDB files.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference PDB file (e.g., 1JV2).",
    )
    parser.add_argument(
        "--trim-json",
        default="storage/avb3_refs/trim_to_reference.json",
        help="Output JSON path for trim ranges.",
    )
    parser.add_argument(
        "--out-dir",
        default="storage/avb3_refs/trimmed_pdb",
        help="Directory for trimmed PDB outputs.",
    )
    return parser.parse_args()


def load_structure(path, pdb_id):
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_id, path)


def ref_residue_map(ref_structure):
    ref = {}
    for model in ref_structure:
        for chain in model:
            residues = set()
            for residue in chain:
                hetflag, resseq, icode = residue.id
                if hetflag.strip():
                    continue
                residues.add((resseq, icode))
            ref[chain.id] = residues
        break
    return ref


def residues_by_chain(structure):
    chain_map = {}
    for model in structure:
        for chain in model:
            residues = []
            for residue in chain:
                hetflag, resseq, icode = residue.id
                if hetflag.strip():
                    continue
                residues.append((resseq, icode))
            chain_map[chain.id] = residues
        break
    return chain_map


def contiguous_ranges(resseqs):
    ranges = []
    if not resseqs:
        return ranges
    start = prev = resseqs[0]
    for num in resseqs[1:]:
        if num == prev + 1:
            prev = num
            continue
        ranges.append([start, prev])
        start = prev = num
    ranges.append([start, prev])
    return ranges


def generate_trim_ranges(ref_map, target_map):
    trim = {}
    for chain_id, residues in target_map.items():
        ref_res = ref_map.get(chain_id, set())
        extra = [r for r, icode in residues if (r, icode) not in ref_res and icode == " "]
        extra = sorted(set(extra))
        if extra:
            trim[chain_id] = contiguous_ranges(extra)
    return trim


def trim_structure_to_reference(structure, ref_map):
    from Bio.PDB.Polypeptide import is_aa

    for model in list(structure):
        for chain in list(model):
            ref_res = ref_map.get(chain.id)
            if ref_res is None:
                model.detach_child(chain.id)
                continue
            for residue in list(chain):
                hetflag, resseq, icode = residue.id
                if hetflag.strip() or not is_aa(residue, standard=False):
                    chain.detach_child(residue.id)
                    continue
                if (resseq, icode) not in ref_res:
                    chain.detach_child(residue.id)
    # Keep only the first model to avoid stray chains in alternate models.
    for model in list(structure)[1:]:
        structure.detach_child(model.id)
    return structure


def save_structure(structure, path):
    from Bio.PDB import PDBIO

    io = PDBIO()
    io.set_structure(structure)
    io.save(path)


def main():
    args = parse_args()
    input_dir = args.input_dir
    ref_path = args.reference

    ref_struct = load_structure(ref_path, "REF")
    ref_map = ref_residue_map(ref_struct)

    trim_config = {}
    os.makedirs(args.out_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".pdb"):
            continue
        path = os.path.join(input_dir, fname)
        pdb_id = os.path.splitext(fname)[0]
        structure = load_structure(path, pdb_id)
        target_map = residues_by_chain(structure)
        trim_ranges = generate_trim_ranges(ref_map, target_map)
        if trim_ranges:
            trim_config[pdb_id] = trim_ranges
        trimmed = trim_structure_to_reference(structure, ref_map)
        out_path = os.path.join(args.out_dir, fname)
        save_structure(trimmed, out_path)

    with open(args.trim_json, "w", encoding="utf-8") as fh:
        json.dump(trim_config, fh, indent=2, sort_keys=True)

    print(f"Wrote trim ranges to {args.trim_json}")
    print(f"Wrote trimmed PDBs to {args.out_dir}")


if __name__ == "__main__":
    main()
