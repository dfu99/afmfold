#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.error
import urllib.request


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def build_query(alpha_accession, beta_accession, rows):
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers."
                            "reference_sequence_identifiers.database_accession"
                        ),
                        "operator": "exact_match",
                        "value": alpha_accession,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers."
                            "reference_sequence_identifiers.database_accession"
                        ),
                        "operator": "exact_match",
                        "value": beta_accession,
                    },
                },
            ],
        },
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
        },
        "return_type": "entry",
    }


def search_rcsb(alpha_accession, beta_accession, rows):
    query = build_query(alpha_accession, beta_accession, rows)
    payload = json.dumps(query).encode()
    req = urllib.request.Request(
        RCSB_SEARCH_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req).read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"RCSB search failed: {exc.code} {detail}") from exc
    result = json.loads(resp)
    return [item["identifier"] for item in result.get("result_set", [])]


def download_mmcif(pdb_id, out_dir):
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    dest = os.path.join(out_dir, f"{pdb_id}.cif")
    urllib.request.urlretrieve(url, dest)
    return dest


def load_trim_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("trim config must be a JSON object keyed by PDB id")
    return {k.upper(): v for k, v in data.items()}


def build_trim_ranges(entry):
    if not entry:
        return {}
    chain_map = {}
    for chain_id, ranges in entry.items():
        parsed = []
        for item in ranges:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(f"range must be [start, end], got {item}")
            start, end = int(item[0]), int(item[1])
            parsed.append((start, end))
        chain_map[chain_id] = parsed
    return chain_map


def residue_in_ranges(resseq, ranges):
    for start, end in ranges:
        if start <= resseq <= end:
            return True
    return False


def strip_structure(mmcif_path, pdb_id, out_path, trim_config):
    try:
        from Bio.PDB import MMCIFParser, PDBIO, Select
        from Bio.PDB.Polypeptide import is_aa
    except ImportError as exc:
        raise RuntimeError(
            "Biopython is required for stripping. Install with: pip install biopython"
        ) from exc

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, mmcif_path)
    chain_trim = build_trim_ranges(trim_config.get(pdb_id.upper()))

    class IntegrinSelect(Select):
        def accept_residue(self, residue):
            if not is_aa(residue, standard=False):
                return False
            if residue.id[0].strip():
                return False
            chain_id = residue.get_parent().id
            ranges = chain_trim.get(chain_id)
            if ranges:
                resseq = residue.id[1]
                if residue_in_ranges(resseq, ranges):
                    return False
            return True

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, IntegrinSelect())


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download integrin heterodimer PDBs and strip to protein-only chains."
        )
    )
    parser.add_argument("--pdb-ids", nargs="*", default=None)
    parser.add_argument(
        "--uniprot-alpha",
        default=None,
        help="Alpha subunit UniProt accession (e.g., P06756 for ITGAV).",
    )
    parser.add_argument(
        "--uniprot-beta",
        default=None,
        help="Beta subunit UniProt accession (e.g., P05106 for ITGB3).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1000,
        help="Maximum number of RCSB entries to return.",
    )
    parser.add_argument(
        "--outdir",
        default="storage/integrin_refs",
        help="Output directory for raw and stripped files.",
    )
    parser.add_argument(
        "--trim-config",
        default=None,
        help=(
            "JSON file with residue ranges to remove per PDB and chain, "
            'e.g. {"4G1E": {"A": [[1, 12]], "B": [[1, 12]]}}.'
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.pdb_ids:
        pdb_ids = [p.upper() for p in args.pdb_ids]
    else:
        if not args.uniprot_alpha or not args.uniprot_beta:
            raise SystemExit(
                "Provide --pdb-ids or both --uniprot-alpha and --uniprot-beta."
            )
        pdb_ids = search_rcsb(args.uniprot_alpha, args.uniprot_beta, args.max_rows)
        if not pdb_ids:
            raise SystemExit("No entries found for those accessions.")

    out_dir = args.outdir
    raw_dir = os.path.join(out_dir, "raw_mmcif")
    stripped_dir = os.path.join(out_dir, "stripped_pdb")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(stripped_dir, exist_ok=True)

    trim_config = load_trim_config(args.trim_config)

    for pdb_id in pdb_ids:
        print(f"Downloading {pdb_id}...")
        mmcif_path = download_mmcif(pdb_id, raw_dir)
        out_path = os.path.join(stripped_dir, f"{pdb_id}.pdb")
        print(f"Stripping {pdb_id} -> {out_path}")
        strip_structure(mmcif_path, pdb_id, out_path, trim_config)

    print(f"Done. Raw mmCIF in {raw_dir}, stripped PDB in {stripped_dir}.")


if __name__ == "__main__":
    main()
