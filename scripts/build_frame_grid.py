#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a frame-by-sample grid image with column headers."
    )
    parser.add_argument(
        "--render-dir",
        required=True,
        help="Directory containing per-frame render subfolders.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output grid PNG path (default: <render-dir>/frame_grid.png).",
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
        help="Regex to extract sample index from PNG filenames.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding in pixels between tiles and around the grid.",
    )
    parser.add_argument(
        "--tile-size",
        default="",
        help="Optional tile size WxH, e.g. 512x512. Empty keeps original size.",
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
    return parser.parse_args()


def parse_tile_size(value):
    if not value:
        return None
    if "x" not in value:
        raise ValueError("tile size must be WxH, e.g. 512x512")
    w_str, h_str = value.split("x", 1)
    return int(w_str), int(h_str)


def pick_default_font_path():
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def measure_text(draw, text, font):
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def frame_sort_key(name):
    match = re.search(r"\d+", name)
    if match:
        return int(match.group(0))
    return name


def collect_samples(frame_dir, sample_re):
    sample_map = {}
    for path in sorted(glob.glob(os.path.join(frame_dir, "*.png"))):
        name = os.path.basename(path)
        match = sample_re.search(name)
        if not match:
            continue
        idx = int(match.group(1))
        if idx not in sample_map:
            sample_map[idx] = path
    return sample_map


def draw_label(draw, text, center_x, center_y, font, fg, bg, pad_x, pad_y):
    text_w, text_h = measure_text(draw, text, font)
    x0 = int(center_x - text_w / 2)
    y0 = int(center_y - text_h / 2)
    if bg.lower() != "none":
        draw.rectangle(
            [x0 - pad_x, y0 - pad_y, x0 + text_w + pad_x, y0 + text_h + pad_y],
            fill=bg,
        )
    draw.text((x0, y0), text, fill=fg, font=font)


def main():
    args = parse_args()
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("Pillow is required: pip install pillow") from exc

    render_dir = os.path.expanduser(args.render_dir)
    if not os.path.isdir(render_dir):
        raise SystemExit(f"Render directory not found: {render_dir}")

    output_path = (
        os.path.expanduser(args.output)
        if args.output
        else os.path.join(render_dir, "frame_grid.png")
    )

    frame_re = re.compile(args.frame_regex)
    frame_names = [
        name
        for name in os.listdir(render_dir)
        if os.path.isdir(os.path.join(render_dir, name)) and frame_re.match(name)
    ]
    if not frame_names:
        raise SystemExit(f"No frame directories matching {args.frame_regex} in {render_dir}")
    frame_names = sorted(frame_names, key=frame_sort_key)

    sample_indices = list(range(args.sample_start, args.sample_start + args.sample_count))
    sample_re = re.compile(args.sample_regex)
    tile_size = parse_tile_size(args.tile_size)
    first_size = None

    images = {}
    missing = {}

    for frame_name in frame_names:
        frame_dir = os.path.join(render_dir, frame_name)
        sample_map = collect_samples(frame_dir, sample_re)
        frame_images = {}
        frame_missing = []
        for sample_idx in sample_indices:
            path = sample_map.get(sample_idx)
            if not path:
                frame_missing.append(sample_idx)
                continue
            img = Image.open(path).convert("RGBA")
            if tile_size:
                img = img.resize(tile_size, resample=Image.LANCZOS)
            else:
                if first_size is None:
                    first_size = img.size
                elif img.size != first_size:
                    img = img.resize(first_size, resample=Image.LANCZOS)
            frame_images[sample_idx] = img
        images[frame_name] = frame_images
        if frame_missing:
            missing[frame_name] = frame_missing

    if tile_size is None:
        if first_size is None:
            raise SystemExit("No PNG images found to determine tile size.")
        tile_size = first_size

    placeholder = Image.new("RGBA", tile_size, (255, 255, 255, 255))

    font_path = args.label_font or pick_default_font_path()
    if font_path:
        font = ImageFont.truetype(font_path, size=args.label_size)
    else:
        font = ImageFont.load_default()

    pad = args.padding
    label_pad_x = max(4, args.label_size // 5)
    label_pad_y = max(3, args.label_size // 6)

    dummy = Image.new("RGB", (10, 10), (255, 255, 255))
    dummy_draw = ImageDraw.Draw(dummy)

    if args.no_column_labels:
        col_header_h = 0
    else:
        col_heights = [
            measure_text(dummy_draw, name, font)[1] for name in frame_names
        ]
        col_header_h = (max(col_heights) if col_heights else 0) + 2 * label_pad_y

    if args.no_row_labels:
        row_header_w = 0
        row_labels = []
    else:
        row_labels = [
            f"{args.row_label_prefix}{idx + args.row_label_offset}"
            for idx in sample_indices
        ]
        row_widths = [measure_text(dummy_draw, label, font)[0] for label in row_labels]
        row_header_w = (max(row_widths) if row_widths else 0) + 2 * label_pad_x

    tile_w, tile_h = tile_size
    cols = len(frame_names)
    rows = len(sample_indices)

    grid_w = row_header_w + cols * tile_w + (cols + 1) * pad
    grid_h = col_header_h + rows * tile_h + (rows + 1) * pad

    grid = Image.new("RGBA", (grid_w, grid_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(grid)

    for col, frame_name in enumerate(frame_names):
        for row, sample_idx in enumerate(sample_indices):
            img = images[frame_name].get(sample_idx, placeholder)
            x0 = row_header_w + pad + col * (tile_w + pad)
            y0 = col_header_h + pad + row * (tile_h + pad)
            grid.paste(img, (x0, y0), img)

    if not args.no_column_labels:
        for col, frame_name in enumerate(frame_names):
            center_x = row_header_w + pad + col * (tile_w + pad) + tile_w / 2
            center_y = col_header_h / 2
            draw_label(
                draw,
                frame_name,
                center_x,
                center_y,
                font,
                args.label_color,
                args.label_bg,
                label_pad_x,
                label_pad_y,
            )

    if not args.no_row_labels:
        for row, label in enumerate(row_labels):
            center_x = row_header_w / 2
            center_y = col_header_h + pad + row * (tile_h + pad) + tile_h / 2
            draw_label(
                draw,
                label,
                center_x,
                center_y,
                font,
                args.label_color,
                args.label_bg,
                label_pad_x,
                label_pad_y,
            )

    grid.convert("RGB").save(output_path)
    print(f"Wrote {output_path}")
    if missing:
        for frame_name, samples in missing.items():
            print(f"[{frame_name}] missing samples: {samples}")


if __name__ == "__main__":
    main()
