#!/usr/bin/env python3
import argparse
import glob
import os
import subprocess
import sys
import tempfile


try:
    from chimerax.core.commands import run
except Exception:  # pragma: no cover - fallback for non-ChimeraX execution
    def _keep_open(argv):
        return "--keep-open-on-error" in argv

    def _relaunch_with_chimerax():
        chimerax_bin = os.environ.get(
            "CHIMERAX_BIN",
            "/Applications/ChimeraX-1.10.1.app/Contents/MacOS/ChimeraX",
        )
        script_path = os.path.abspath(__file__)
        temp_dir = tempfile.mkdtemp(prefix="chimerax_")
        safe_script = os.path.join(temp_dir, "chimerax_render_pdb_images.py")
        try:
            os.symlink(script_path, safe_script)
        except FileExistsError:
            pass
        cmd = [chimerax_bin, "--script", safe_script, "--"]
        if not _keep_open(sys.argv[1:]):
            cmd.insert(1, "--exit")
        cmd.extend(sys.argv[1:])
        subprocess.run(cmd, check=True)

    if __name__ == "__main__":
        _relaunch_with_chimerax()
        sys.exit(0)
    raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render per-PDB PNGs in ChimeraX for external grid assembly."
    )
    parser.add_argument(
        "--input-dir",
        default="storage/avb3_refs/trimmed_pdb",
        help="Directory containing stripped PDBs.",
    )
    parser.add_argument(
        "--out-dir",
        default="storage/avb3_refs/trimmed_renders",
        help="Directory to write PNG stripped renders.",
    )
    parser.add_argument(
        "--supersample",
        type=int,
        default=3,
        help="Supersample factor for image quality.",
    )
    parser.add_argument(
        "--keep-open-on-error",
        action="store_true",
        help="Keep ChimeraX open if the script fails.",
    )
    return parser.parse_args()


def main(session):
    args = parse_args()
    input_dir = args.input_dir
    out_dir = args.out_dir
    pdb_paths = sorted(glob.glob(os.path.join(input_dir, "*.pdb")))
    if not pdb_paths:
        raise SystemExit(f"No PDB files found in {input_dir}")
    try:
        os.makedirs(out_dir, exist_ok=True)
        print(f"Output directory: {out_dir}")
    except Exception as e:
        raise SystemExit(f"Failed to create output directory {out_dir}: {e}")

    for pdb_path in pdb_paths:
        run(session, f'open "{pdb_path}"')
        run(session, "color bychain")
        run(session, "set bgColor white")
        run(session, "lighting soft")
        run(session, "view")
        stem = os.path.splitext(os.path.basename(pdb_path))[0]
        out_path = os.path.join(out_dir, f"{stem}.png")
        run(session, f'save "{out_path}" supersample {args.supersample}')
        run(session, "close all")


try:
    main(session)
except Exception:
    import traceback

    traceback.print_exc()
    try:
        session.logger.error("ChimeraX render failed; see traceback above.")
    except Exception:
        pass
    raise
