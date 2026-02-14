#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import shlex
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

        env = os.environ.copy()
        for key in list(env):
            if key.startswith("CONDA"):
                env.pop(key, None)
        for key in [
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONUSERBASE",
            "PYTHONEXECUTABLE",
            "PYTHONSTARTUP",
            "PYTHONWARNINGS",
            "PYTHONNOUSERSITE",
            "VIRTUAL_ENV",
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
            "DYLD_FALLBACK_LIBRARY_PATH",
        ]:
            env.pop(key, None)
        env["PYTHONNOUSERSITE"] = "1"
        script_path = os.path.abspath(__file__)
        temp_dir = tempfile.mkdtemp(prefix="chimerax_")
        safe_script = os.path.join(temp_dir, "chimerax_render_frame_cif_images.py")
        try:
            os.symlink(script_path, safe_script)
        except FileExistsError:
            pass
        env["CHIMERAX_SCRIPT_ARGS"] = json.dumps(sys.argv[1:])
        cmd = [chimerax_bin, "--script", safe_script]
        if not _keep_open(sys.argv[1:]):
            cmd.insert(1, "--exit")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print("ChimeraX invocation failed.")
            print("Command:", " ".join(cmd))
            print("Note: cleared CONDA/PYTHON env vars for ChimeraX isolation.")
            if result.stdout:
                print("--- ChimeraX stdout ---")
                print(result.stdout)
            if result.stderr:
                print("--- ChimeraX stderr ---")
                print(result.stderr)
            raise SystemExit(
                "ChimeraX exited with a non-zero status. "
                "Verify the ChimeraX path or set CHIMERAX_BIN."
            )

    if __name__ == "__main__":
        _relaunch_with_chimerax()
        sys.exit(0)
    raise


def _apply_env_args():
    raw = os.environ.get("CHIMERAX_SCRIPT_ARGS")
    if not raw:
        return
    try:
        extra = json.loads(raw)
    except json.JSONDecodeError:
        extra = shlex.split(raw)
    if isinstance(extra, list):
        sys.argv = [sys.argv[0]] + [str(arg) for arg in extra]


def parse_args():
    _apply_env_args()
    parser = argparse.ArgumentParser(
        description="Render CIF snapshots per frame in ChimeraX."
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory containing frame_* folders.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory for PNG renders (default: <root>/renders).",
    )
    parser.add_argument(
        "--frame-regex",
        default=r"^frame_\d+$",
        help="Regex for frame folder names.",
    )
    parser.add_argument(
        "--predictions-subdir",
        default="production_trajector/seed_101/predictions",
        help="Relative path from frame folder to predictions directory.",
    )
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="Starting sample index to render.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="Number of samples per frame to render.",
    )
    parser.add_argument(
        "--sample-regex",
        default=r"sample[_-]?(\d+)",
        help="Regex to extract sample index from CIF filenames.",
    )
    parser.add_argument(
        "--supersample",
        type=int,
        default=3,
        help="Supersample factor for image quality.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional path to write a JSON manifest (default: <out_dir>/render_manifest.json).",
    )
    parser.add_argument(
        "--keep-open-on-error",
        action="store_true",
        help="Keep ChimeraX open if the script fails.",
    )
    return parser.parse_args()


def _frame_sort_key(name):
    match = re.search(r"\d+", name)
    if match:
        return int(match.group(0))
    return name


def _find_frame_dirs(root_dir, frame_re):
    entries = []
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path) and frame_re.match(name):
            entries.append(name)
    return sorted(entries, key=_frame_sort_key)


def _resolve_predictions_dir(frame_dir, predictions_subdir):
    candidate = os.path.join(frame_dir, predictions_subdir)
    if os.path.isdir(candidate):
        return candidate
    matches = [
        path
        for path in glob.glob(os.path.join(frame_dir, "**", "predictions"), recursive=True)
        if os.path.isdir(path)
    ]
    if not matches:
        raise SystemExit(f"No predictions directory found in {frame_dir}")
    if len(matches) > 1:
        joined = "\n".join(matches)
        raise SystemExit(
            "Multiple predictions directories found; use --predictions-subdir to pick one:\n"
            f"{joined}"
        )
    return matches[0]


def _collect_samples(predictions_dir, sample_re):
    sample_map = {}
    for path in sorted(glob.glob(os.path.join(predictions_dir, "*.cif"))):
        name = os.path.basename(path)
        match = sample_re.search(name)
        if not match:
            continue
        idx = int(match.group(1))
        if idx not in sample_map:
            sample_map[idx] = path
    return sample_map


def main(session):
    args = parse_args()
    root_dir = os.path.expanduser(args.root_dir)
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else os.path.join(root_dir, "renders")
    frame_re = re.compile(args.frame_regex)
    sample_re = re.compile(args.sample_regex)
    sample_indices = list(range(args.sample_start, args.sample_start + args.sample_count))

    if not os.path.isdir(root_dir):
        raise SystemExit(f"Root directory not found: {root_dir}")

    frame_names = _find_frame_dirs(root_dir, frame_re)
    if not frame_names:
        raise SystemExit(f"No frame directories matching {args.frame_regex} in {root_dir}")

    os.makedirs(out_dir, exist_ok=True)
    manifest_path = (
        os.path.expanduser(args.manifest)
        if args.manifest
        else os.path.join(out_dir, "render_manifest.json")
    )
    manifest = {
        "root_dir": root_dir,
        "out_dir": out_dir,
        "frame_regex": args.frame_regex,
        "predictions_subdir": args.predictions_subdir,
        "sample_start": args.sample_start,
        "sample_count": args.sample_count,
        "frames": [],
    }

    for frame_name in frame_names:
        frame_dir = os.path.join(root_dir, frame_name)
        predictions_dir = _resolve_predictions_dir(frame_dir, args.predictions_subdir)
        sample_map = _collect_samples(predictions_dir, sample_re)
        missing = []
        rendered = {}

        out_frame_dir = os.path.join(out_dir, frame_name)
        os.makedirs(out_frame_dir, exist_ok=True)

        for sample_idx in sample_indices:
            cif_path = sample_map.get(sample_idx)
            if not cif_path:
                missing.append(sample_idx)
                continue
            stem = os.path.splitext(os.path.basename(cif_path))[0]
            out_path = os.path.join(out_frame_dir, f"{stem}.png")
            run(session, f'open "{cif_path}"')
            run(session, "color bychain")
            run(session, "set bgColor white")
            run(session, "lighting soft")
            run(session, "view")
            run(session, f'save "{out_path}" supersample {args.supersample}')
            run(session, "close all")
            rendered[str(sample_idx)] = {"cif": cif_path, "png": out_path}

        if missing:
            print(f"[{frame_name}] missing samples: {missing}")

        manifest["frames"].append(
            {
                "name": frame_name,
                "frame_dir": frame_dir,
                "predictions_dir": predictions_dir,
                "samples": rendered,
                "missing": missing,
            }
        )

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Rendered frames to: {out_dir}")


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
