#!/usr/bin/env python3
"""Interactive polygon-marking tool for shelf and surface regions.

Usage:
    python tools/mark_shelf_regions.py --image data/reference_frames/store_camera_01_reference.jpg

Controls:
    Left-click        Add a vertex to the current polygon
    Right-click / U   Undo the last vertex
    Enter             Finish polygon (prompts for region_id and type)
    Escape            Cancel the polygon in progress
    S                 Save all regions to the output YAML
    Q                 Quit (prompts to save if there are unsaved regions)
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog
except ImportError:
    sys.exit("tkinter is required. Install it via your OS package manager.")

try:
    from PIL import Image, ImageTk
except ImportError:
    sys.exit("Pillow is required: pip install 'pickup-putdown[viz]'")

try:
    import typer
except ImportError:
    sys.exit("typer is required: pip install typer")

app = typer.Typer(add_completion=False)

# Maximum canvas size; image is scaled down to fit if larger.
_MAX_W, _MAX_H = 1400, 880

_VERTEX_RADIUS = 5
_ACTIVE_COLOR = "#FF4444"  # in-progress polygon vertices and edges
_SAVED_COLOR = "#22CC44"  # finalized polygon outline and label
_VERTEX_OUTLINE = "#FFFFFF"


class MarkingApp:
    def __init__(
        self,
        root: tk.Tk,
        image_path: Path,
        camera_id: str,
        expansion_mode: str,
        expansion_value: float,
        output_path: Path,
    ) -> None:
        self.root = root
        self.camera_id = camera_id
        self.expansion_mode = expansion_mode
        self.expansion_value = expansion_value
        self.output_path = output_path

        # Load image and compute display scale.
        self.orig_image = Image.open(image_path).convert("RGB")
        self.source_w, self.source_h = self.orig_image.size
        self.scale = min(_MAX_W / self.source_w, _MAX_H / self.source_h, 1.0)
        disp_w = int(self.source_w * self.scale)
        disp_h = int(self.source_h * self.scale)

        # Keep reference so GC doesn't collect it.
        self._tk_image = ImageTk.PhotoImage(
            self.orig_image.resize((disp_w, disp_h), Image.LANCZOS)
        )

        # State
        self.current_pts: list[tuple[float, float]] = []  # source-space coords
        self.saved_regions: list[dict] = []
        self._canvas_items: list[int] = []  # items for the in-progress polygon

        # Load existing regions from file so they are shown at startup.
        self._load_existing_regions()

        # Build UI
        root.title(f"Mark shelf regions — {camera_id}")
        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(frame, width=disp_w, height=disp_h, cursor="crosshair")
        self.canvas.pack(side=tk.TOP)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_image)

        self.status_var = tk.StringVar(
            value="Left-click: add point | Enter: finish polygon | U: undo | S: save | Q: quit"
        )
        tk.Label(root, textvariable=self.status_var, anchor="w", bg="#222", fg="#EEE").pack(
            fill=tk.X, side=tk.BOTTOM
        )

        # Draw already-loaded regions.
        for region in self.saved_regions:
            self._draw_finalized_region(region["region_id"], region["polygon"])

        # Bind events
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_undo)
        root.bind("<Return>", self._on_enter)
        root.bind("<KP_Enter>", self._on_enter)
        root.bind("<u>", self._on_undo)
        root.bind("<U>", self._on_undo)
        root.bind("<s>", self._on_save_key)
        root.bind("<S>", self._on_save_key)
        root.bind("<q>", self._on_quit)
        root.bind("<Q>", self._on_quit)
        root.bind("<Escape>", self._on_cancel)
        root.protocol("WM_DELETE_WINDOW", self._on_quit)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_left_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        src_x = event.x / self.scale
        src_y = event.y / self.scale
        self.current_pts.append((src_x, src_y))
        self._redraw_current()
        n = len(self.current_pts)
        self.status_var.set(f"{n} point(s) | Enter: finish | U: undo | Esc: cancel")

    def _on_undo(self, _event: tk.Event | None = None) -> None:
        if self.current_pts:
            self.current_pts.pop()
            self._redraw_current()
            self.status_var.set(f"{len(self.current_pts)} point(s) | U: undo again")

    def _on_enter(self, _event: tk.Event | None = None) -> None:
        if len(self.current_pts) < 3:
            messagebox.showwarning(
                "Too few points",
                f"Need at least 3 points (have {len(self.current_pts)}).",
                parent=self.root,
            )
            return
        region_id = simpledialog.askstring(
            "Region ID",
            "Enter a stable region_id (e.g. shelf_01):",
            parent=self.root,
        )
        if not region_id:
            return
        region_id = region_id.strip()

        region_type = simpledialog.askstring(
            "Region type",
            "Type (shelf / surface):",
            initialvalue="shelf",
            parent=self.root,
        )
        if not region_type or region_type.strip() not in ("shelf", "surface"):
            messagebox.showerror(
                "Invalid type", "Type must be 'shelf' or 'surface'.", parent=self.root
            )
            return

        polygon = [[round(x), round(y)] for x, y in self.current_pts]
        self.saved_regions.append(
            {"region_id": region_id, "type": region_type.strip(), "polygon": polygon}
        )
        self._draw_finalized_region(region_id, polygon)

        # Clear in-progress state.
        for item in self._canvas_items:
            self.canvas.delete(item)
        self._canvas_items = []
        self.current_pts = []
        self.status_var.set(
            f"Saved '{region_id}'. Total regions: {len(self.saved_regions)}. "
            "Click to start next polygon, S to save file."
        )

    def _on_cancel(self, _event: tk.Event | None = None) -> None:
        for item in self._canvas_items:
            self.canvas.delete(item)
        self._canvas_items = []
        self.current_pts = []
        self.status_var.set("Polygon cancelled. Left-click to start a new one.")

    def _on_save_key(self, _event: tk.Event | None = None) -> None:
        self._save()

    def _on_quit(self, _event: tk.Event | None = None) -> None:
        if self.current_pts and not messagebox.askyesno(
            "Discard polygon?",
            "There is an unfinished polygon. Quit anyway?",
            parent=self.root,
        ):
            return
        if self.saved_regions and messagebox.askyesno(
            "Save before quitting?",
            f"Save {len(self.saved_regions)} region(s) to {self.output_path}?",
            parent=self.root,
        ):
            self._save()
        self.root.destroy()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _redraw_current(self) -> None:
        for item in self._canvas_items:
            self.canvas.delete(item)
        self._canvas_items = []

        for i, (x, y) in enumerate(self.current_pts):
            cx, cy = x * self.scale, y * self.scale
            r = _VERTEX_RADIUS
            self._canvas_items.append(
                self.canvas.create_oval(
                    cx - r,
                    cy - r,
                    cx + r,
                    cy + r,
                    fill=_ACTIVE_COLOR,
                    outline=_VERTEX_OUTLINE,
                )
            )
            if i > 0:
                px, py = self.current_pts[i - 1]
                self._canvas_items.append(
                    self.canvas.create_line(
                        px * self.scale,
                        py * self.scale,
                        cx,
                        cy,
                        fill=_ACTIVE_COLOR,
                        width=2,
                    )
                )

    def _draw_finalized_region(self, region_id: str, polygon: list[list[int]]) -> None:
        disp_pts = [(p[0] * self.scale, p[1] * self.scale) for p in polygon]
        flat = [coord for pt in disp_pts for coord in pt]
        self.canvas.create_polygon(*flat, outline=_SAVED_COLOR, fill="", width=2)
        cx = sum(p[0] for p in disp_pts) / len(disp_pts)
        cy = sum(p[1] for p in disp_pts) / len(disp_pts)
        self.canvas.create_text(
            cx,
            cy,
            text=region_id,
            fill=_SAVED_COLOR,
            font=("Helvetica", 11, "bold"),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_existing_regions(self) -> None:
        """Populate self.saved_regions from an existing shelves.yaml if present."""
        if not self.output_path.exists():
            return
        try:
            with self.output_path.open() as fh:
                raw = yaml.safe_load(fh) or {}
            cam = (raw.get("cameras") or {}).get(self.camera_id)
            if cam and isinstance(cam.get("regions"), list):
                self.saved_regions = list(cam["regions"])
        except Exception:
            pass  # Corrupt file — start fresh for this camera.

    def _save(self) -> None:
        """Write all marked regions for this camera to the output YAML."""
        existing: dict = {}
        if self.output_path.exists():
            try:
                with self.output_path.open() as fh:
                    existing = yaml.safe_load(fh) or {}
            except Exception:
                pass

        if "cameras" not in existing or not isinstance(existing["cameras"], dict):
            existing["cameras"] = {}

        existing["cameras"][self.camera_id] = {
            "source_width": self.source_w,
            "source_height": self.source_h,
            "expansion": {"mode": self.expansion_mode, "value": self.expansion_value},
            "regions": self.saved_regions,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a custom dumper so each [x, y] pair is written in flow style.
        _dump_shelves_yaml(existing, self.output_path)
        self.status_var.set(f"Saved {len(self.saved_regions)} region(s) to {self.output_path}")


# ---------------------------------------------------------------------------
# YAML serialisation helpers
# ---------------------------------------------------------------------------


class _FlowList(list):
    """Marker subclass so PyYAML outputs this list in flow style."""


def _flow_representer(dumper: yaml.Dumper, data: _FlowList) -> yaml.Node:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _build_camera_dict(data: dict) -> dict:
    """Replace plain polygon point lists with _FlowList so they dump as [x, y]."""
    result: dict = {}
    for cam_id, cam_val in data.get("cameras", {}).items():
        cam_copy: dict = dict(cam_val)
        regions = []
        for region in cam_copy.get("regions", []):
            r = dict(region)
            r["polygon"] = [_FlowList(pt) for pt in r.get("polygon", [])]
            regions.append(r)
        cam_copy["regions"] = regions
        result[cam_id] = cam_copy
    return {"cameras": result}


def _dump_shelves_yaml(data: dict, path: Path) -> None:
    dumper = yaml.Dumper
    yaml.add_representer(_FlowList, _flow_representer, Dumper=dumper)
    structured = _build_camera_dict(data)
    with path.open("w") as fh:
        yaml.dump(structured, fh, Dumper=dumper, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    image: Path = typer.Option(  # noqa: B008
        Path("data/reference_frames/store_camera_01_reference.jpg"),
        "--image",
        "-i",
        help="Path to the reference frame.",
    ),
    camera_id: str = typer.Option(  # noqa: B008
        "store_camera_01", "--camera-id", help="Camera identifier written to YAML."
    ),
    output: Path = typer.Option(  # noqa: B008
        Path("configs/shelves.yaml"),
        "--output",
        "-o",
        help="Output shelves YAML file.",
    ),
    expansion_mode: str = typer.Option(  # noqa: B008
        "pixel", "--expansion-mode", help="pixel or normalized."
    ),
    expansion_value: float = typer.Option(  # noqa: B008
        20.0, "--expansion-value", help="Expansion margin (pixels, or fraction for normalized)."
    ),
) -> None:
    """Interactively mark shelf/surface polygons on a reference frame."""
    if not image.exists():
        typer.echo(f"Error: image not found: {image}", err=True)
        raise typer.Exit(1)
    if expansion_mode not in ("pixel", "normalized"):
        typer.echo("Error: --expansion-mode must be 'pixel' or 'normalized'.", err=True)
        raise typer.Exit(1)

    root = tk.Tk()
    MarkingApp(
        root=root,
        image_path=image,
        camera_id=camera_id,
        expansion_mode=expansion_mode,
        expansion_value=expansion_value,
        output_path=output,
    )
    root.mainloop()


if __name__ == "__main__":
    app()
