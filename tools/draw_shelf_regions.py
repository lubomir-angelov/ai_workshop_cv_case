#!/usr/bin/env python3
"""Render shelf and surface region overlays onto a reference frame.

Produces a single PNG showing:
  - Exact region polygons (blue, semi-transparent fill)
  - Expanded interaction-zone polygons (orange, semi-transparent fill)
  - Region ID labels at each polygon centroid

Usage:
    python tools/draw_shelf_regions.py \\
        --image  data/reference_frames/store_camera_01_reference.jpg \\
        --config configs/shelves.yaml \\
        --camera-id store_camera_01 \\
        --output data/overlays/store_camera_01_shelf_overlay.png
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Pillow is required: pip install 'pickup-putdown[viz]'")

try:
    import typer
except ImportError:
    sys.exit("typer is required: pip install typer")

# Add src/ to the path so the tool works when run from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pickup_putdown.perception.shelf_regions import (
    get_expanded_regions,
    get_regions_for_camera,
    load_shelf_config,
)

log = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

# Overlay colours (R, G, B, A)
_EXACT_FILL = (59, 130, 246, 80)  # blue, ~30 % opacity
_EXACT_OUTLINE = (59, 130, 246, 230)
_EXP_FILL = (245, 158, 11, 55)  # amber, ~20 % opacity
_EXP_OUTLINE = (245, 158, 11, 200)
_LABEL_FILL = (255, 255, 255, 230)
_LABEL_SHADOW = (0, 0, 0, 200)

_OUTLINE_WIDTH = 3


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _to_int_pts(
    pts: list[tuple[float, float]],
) -> list[tuple[int, int]]:
    return [(int(round(x)), int(round(y))) for x, y in pts]


def _centroid(pts: list[tuple[float, float]]) -> tuple[int, int]:
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return int(round(cx)), int(round(cy))


def draw_overlay(
    ref_image: Image.Image,
    camera_config,  # CameraShelfConfig
    font_size: int = 40,
) -> Image.Image:
    """Return a copy of *ref_image* with region polygons composited over it."""
    base = ref_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(font_size)

    expanded = get_expanded_regions(camera_config)

    for region in camera_config.regions:
        rid = region.region_id
        exact_pts = _to_int_pts(region.points)
        exp_pts = _to_int_pts(expanded[rid])

        # Expanded interaction zone — drawn first so exact sits on top.
        draw.polygon(exp_pts, fill=_EXP_FILL, outline=None)
        for i in range(len(exp_pts)):
            a = exp_pts[i]
            b = exp_pts[(i + 1) % len(exp_pts)]
            draw.line([a, b], fill=_EXP_OUTLINE, width=_OUTLINE_WIDTH)

        # Exact shelf/surface region.
        draw.polygon(exact_pts, fill=_EXACT_FILL, outline=None)
        for i in range(len(exact_pts)):
            a = exact_pts[i]
            b = exact_pts[(i + 1) % len(exact_pts)]
            draw.line([a, b], fill=_EXACT_OUTLINE, width=_OUTLINE_WIDTH)

        # Label at centroid.
        cx, cy = _centroid(region.points)
        # Shadow
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            draw.text((cx + dx, cy + dy), rid, font=font, fill=_LABEL_SHADOW, anchor="mm")
        draw.text((cx, cy), rid, font=font, fill=_LABEL_FILL, anchor="mm")

    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    image: Path = typer.Option(  # noqa: B008
        Path("data/reference_frames/store_camera_01_reference.jpg"),
        "--image",
        "-i",
        help="Reference frame to annotate.",
    ),
    config: Path = typer.Option(  # noqa: B008
        Path("configs/shelves.yaml"),
        "--config",
        "-c",
        help="Shelf configuration YAML.",
    ),
    camera_id: str | None = typer.Option(  # noqa: B008
        None, "--camera-id", help="Camera to render (default: first camera in config)."
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Output image path (default: data/overlays/<camera>_shelf_overlay.png).",
    ),
    font_size: int = typer.Option(40, "--font-size", help="Label font size in pixels."),
) -> None:
    """Generate a shelf region overlay image for visual validation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not image.exists():
        typer.echo(f"Error: image not found: {image}", err=True)
        raise typer.Exit(1)

    shelf_cfg = load_shelf_config(config)

    resolved_camera_id = camera_id or next(iter(shelf_cfg.cameras))
    camera_cfg = get_regions_for_camera(shelf_cfg, resolved_camera_id)

    typer.echo(
        f"Camera: {resolved_camera_id} | "
        f"{len(camera_cfg.regions)} region(s) | "
        f"source {camera_cfg.source_width}x{camera_cfg.source_height}"
    )

    ref_img = Image.open(image)
    if (ref_img.width, ref_img.height) != (camera_cfg.source_width, camera_cfg.source_height):
        typer.echo(
            f"Warning: image size {ref_img.width}x{ref_img.height} does not match "
            f"config source resolution {camera_cfg.source_width}x{camera_cfg.source_height}.",
            err=True,
        )

    result = draw_overlay(ref_img, camera_cfg, font_size=font_size)

    if output is None:
        out_dir = Path("data/overlays")
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"{resolved_camera_id}_shelf_overlay.png"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    result.save(str(output))
    typer.echo(f"Overlay saved: {output}")


if __name__ == "__main__":
    app()
