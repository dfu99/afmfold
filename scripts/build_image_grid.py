#!/usr/bin/env python3
import argparse
import glob
import math
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a labeled grid image from per-PDB renders."
    )
    parser.add_argument(
        "--input-dir",
        default="storage/avb3_refs/renders",
        help="Directory containing PNG renders.",
    )
    parser.add_argument(
        "--output",
        default="storage/avb3_refs/avb3_grid.png",
        help="Output grid PNG path.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=0,
        help="Number of columns; defaults to sqrt(N).",
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
        "--label-offset",
        type=int,
        default=8,
        help="Label offset from the top-left of each tile in pixels.",
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


def main():
    args = parse_args()
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise SystemExit("Pillow is required: pip install pillow") from exc

    input_dir = args.input_dir
    paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not paths:
        raise SystemExit(f"No PNG files found in {input_dir}")

    tile_size = parse_tile_size(args.tile_size)
    images = []
    labels = []
    for path in paths:
        img = Image.open(path).convert("RGBA")
        if tile_size:
            img = img.resize(tile_size, resample=Image.LANCZOS)
        images.append(img)
        labels.append(os.path.splitext(os.path.basename(path))[0])

    if not images:
        raise SystemExit("No images loaded")

    tile_w, tile_h = images[0].size
    count = len(images)
    cols = args.columns or int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / cols))
    pad = args.padding

    grid_w = cols * tile_w + (cols + 1) * pad
    grid_h = rows * tile_h + (rows + 1) * pad
    grid = Image.new("RGBA", (grid_w, grid_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(grid)

    font_path = args.label_font or pick_default_font_path()
    if font_path:
        font = ImageFont.truetype(font_path, size=args.label_size)
    else:
        font = ImageFont.load_default()

    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * (tile_w + pad)
        y0 = pad + row * (tile_h + pad)
        grid.paste(img, (x0, y0), img)

        lx = x0 + args.label_offset
        ly = y0 + args.label_offset
        if args.label_bg.lower() != "none":
            text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
            bg = Image.new("RGBA", (text_w + 6, text_h + 4), args.label_bg)
            grid.paste(bg, (lx - 3, ly - 2), bg)
        draw.text((lx, ly), label, fill=args.label_color, font=font)

    grid.convert("RGB").save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
