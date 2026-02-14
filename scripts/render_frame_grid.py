#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ChimeraX rendering and build a frame-by-sample grid."
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory containing frame_* folders.",
    )
    parser.add_argument(
        "--render-dir",
        default="",
        help="Output directory for PNG renders (default: <root>/renders).",
    )
    parser.add_argument(
        "--grid-output",
        default="",
        help="Output grid PNG path (default: <render-dir>/frame_grid.png).",
    )
    parser.add_argument(
        "--predictions-subdir",
        default="production_trajector/seed_101/predictions",
        help="Relative path from frame folder to predictions directory.",
    )
    parser.add_argument(
        "--frame-regex",
        default=r"^frame_\d+$",
        help="Regex for frame folder names.",
    )
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="Starting sample index.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="Number of sample rows.",
    )
    parser.add_argument(
        "--sample-regex",
        default=r"sample[_-]?(\d+)",
        help="Regex to extract sample index from filenames.",
    )
    parser.add_argument(
        "--supersample",
        type=int,
        default=3,
        help="Supersample factor for image quality.",
    )
    parser.add_argument(
        "--tile-size",
        default="",
        help="Optional tile size WxH, e.g. 512x512. Empty keeps original size.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding in pixels between tiles and around the grid.",
    )
    parser.add_argument(
        "--label-color",
        default="black",
        help="Label text color.",
    )
    parser.add_argument(
        "--label-bg",
        default="white",
        help="Label background color or 'none'.",
    )
    parser.add_argument(
        "--label-font",
        default="",
        help="Optional path to a TTF font file.",
    )
    parser.add_argument(
        "--label-size",
        type=int,
        default=20,
        help="Label font size (TTF only).",
    )
    parser.add_argument(
        "--row-label-prefix",
        default="sample ",
        help="Row label prefix.",
    )
    parser.add_argument(
        "--row-label-offset",
        type=int,
        default=1,
        help="Offset added to the sample index for row labels.",
    )
    parser.add_argument(
        "--no-row-labels",
        action="store_true",
        help="Disable row labels.",
    )
    parser.add_argument(
        "--no-column-labels",
        action="store_true",
        help="Disable column headers.",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Skip ChimeraX rendering step.",
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="Skip grid-building step.",
    )
    parser.add_argument(
        "--keep-open-on-error",
        action="store_true",
        help="Keep ChimeraX open if the render script fails.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = os.path.expanduser(args.root_dir)
    render_dir = os.path.expanduser(args.render_dir) if args.render_dir else os.path.join(root_dir, "renders")
    grid_output = (
        os.path.expanduser(args.grid_output)
        if args.grid_output
        else os.path.join(render_dir, "frame_grid.png")
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    render_script = os.path.join(script_dir, "chimerax_render_frame_cif_images.py")
    grid_script = os.path.join(script_dir, "build_frame_grid.py")

    if not args.skip_render:
        cmd = [
            sys.executable,
            render_script,
            "--root-dir",
            root_dir,
            "--out-dir",
            render_dir,
            "--predictions-subdir",
            args.predictions_subdir,
            "--frame-regex",
            args.frame_regex,
            "--sample-start",
            str(args.sample_start),
            "--sample-count",
            str(args.sample_count),
            "--sample-regex",
            args.sample_regex,
            "--supersample",
            str(args.supersample),
        ]
        if args.keep_open_on_error:
            cmd.append("--keep-open-on-error")
        subprocess.run(cmd, check=True)

    if not args.skip_grid:
        cmd = [
            sys.executable,
            grid_script,
            "--render-dir",
            render_dir,
            "--output",
            grid_output,
            "--frame-regex",
            args.frame_regex,
            "--sample-start",
            str(args.sample_start),
            "--sample-count",
            str(args.sample_count),
            "--sample-regex",
            args.sample_regex,
            "--tile-size",
            args.tile_size,
            "--padding",
            str(args.padding),
            "--label-color",
            args.label_color,
            "--label-bg",
            args.label_bg,
            "--label-font",
            args.label_font,
            "--label-size",
            str(args.label_size),
            "--row-label-prefix",
            args.row_label_prefix,
            "--row-label-offset",
            str(args.row_label_offset),
        ]
        if args.no_row_labels:
            cmd.append("--no-row-labels")
        if args.no_column_labels:
            cmd.append("--no-column-labels")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
