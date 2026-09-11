#!/usr/bin/env python3
"""Build the Track B1 conference presentation.

    python scripts/make_track_b1_slides.py --output .local/paper/track_b1_presentation.pptx

A scientific-talk deck (~20 min) covering the Track B1 pickup/putdown detector:
problem framing, data and annotation, pipeline, the two methodological findings,
results, and the cross-day generalisation failure.

Every number is read from the stored metrics artefacts where one exists, so a
re-run of the pipeline updates the slides. Speaker notes are attached to each slide.

Palette: white ground, dark purple accents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------

PURPLE_DEEP = RGBColor(0x3D, 0x20, 0x63)   # headline / title grounds
PURPLE_MID = RGBColor(0x6B, 0x3F, 0xA0)    # accents, rules, emphasis
PURPLE_TINT = RGBColor(0xF3, 0xEE, 0xF9)   # panel fills
PURPLE_EDGE = RGBColor(0xD9, 0xCB, 0xEC)   # panel borders
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1F, 0x1F, 0x1F)
MUTED = RGBColor(0x6B, 0x6B, 0x78)

FONT = "Arial"
W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.85)
BODY_W = W - 2 * MARGIN


def _text(frame, runs, *, size, color, bold=False, space_after=8, line=1.25, align=PP_ALIGN.LEFT):
    """Write paragraphs into a text frame. `runs` is a list of strings or (text, bold) pairs."""
    frame.word_wrap = True
    for index, item in enumerate(runs):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space_after)
        para.line_spacing = line
        if isinstance(item, tuple):
            body, is_bold = item
        else:
            body, is_bold = item, bold
        run = para.add_run()
        run.text = body
        run.font.size = Pt(size)
        run.font.bold = is_bold
        run.font.color.rgb = color
        run.font.name = FONT
    return frame


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def fill(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def box(slide, left, top, width, height):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tb.text_frame.word_wrap = True
    return tb.text_frame


def rule(slide, top, left=MARGIN, width=Inches(1.6), height=Pt(4), color=PURPLE_MID):
    from pptx.enum.shapes import MSO_SHAPE
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def panel(slide, left, top, width, height, fill_color=PURPLE_TINT, edge=PURPLE_EDGE):
    from pptx.enum.shapes import MSO_SHAPE
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.adjustments[0] = 0.04
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.color.rgb = edge
    shape.line.width = Pt(1)
    shape.shadow.inherit = False
    return shape


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


TITLE_PT = 30


def heading(slide, title, kicker=None):
    """Standard content-slide header: kicker, rule, title.

    Returns the y offset where body content may start. A title that wraps to a second
    line pushes the body down rather than colliding with it.
    """
    top = Inches(0.55)
    if kicker:
        _text(box(slide, MARGIN, top, BODY_W, Inches(0.32)), [kicker.upper()],
              size=12, color=PURPLE_MID, bold=True, space_after=0)
        top = top + Inches(0.36)
    rule(slide, top + Inches(0.04))

    # ~0.50 * point size is a serviceable average glyph width for Arial bold.
    chars_per_line = max(1, int((BODY_W / 914400) * 72 / (TITLE_PT * 0.50)))
    lines = max(1, -(-len(title) // chars_per_line))
    title_h = Inches(lines * TITLE_PT * 1.05 / 72)

    _text(box(slide, MARGIN, top + Inches(0.18), BODY_W, title_h + Inches(0.1)), [title],
          size=TITLE_PT, color=PURPLE_DEEP, bold=True, space_after=0, line=1.05)
    return top + Inches(0.18) + title_h + Inches(0.32)


def bullets(slide, items, top, *, size=17, width=None, left=MARGIN):
    """Bulleted body text. Items may be (text, bold) pairs."""
    frame = box(slide, left, top, width or BODY_W, H - top - Inches(0.7))
    rendered = []
    for item in items:
        if isinstance(item, tuple):
            rendered.append(("•  " + item[0], item[1]))
        else:
            rendered.append("•  " + item)
    return _text(frame, rendered, size=size, color=INK, space_after=13, line=1.28)


def stat_panel(slide, left, top, width, value, label, height=Inches(1.45)):
    panel(slide, left, top, width, height)
    _text(box(slide, left + Inches(0.2), top + Inches(0.16), width - Inches(0.4), Inches(0.6)),
          [value], size=30, color=PURPLE_DEEP, bold=True, space_after=0, align=PP_ALIGN.CENTER)
    _text(box(slide, left + Inches(0.2), top + Inches(0.83), width - Inches(0.4), Inches(0.5)),
          [label], size=12, color=MUTED, space_after=0, align=PP_ALIGN.CENTER, line=1.1)


def table(slide, rows, left, top, width, col_widths=None, size=13, header=True):
    n_rows, n_cols = len(rows), len(rows[0])
    height = Inches(0.4) * n_rows
    shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    tbl = shape.table
    if col_widths:
        total = sum(col_widths)
        for i, frac in enumerate(col_widths):
            tbl.columns[i].width = Emu(int(width * frac / total))
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(0.38)
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = ""
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left, cell.margin_right = Inches(0.12), Inches(0.12)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.fill.solid()
            if header and r == 0:
                cell.fill.fore_color.rgb = PURPLE_DEEP
                colour, bold = WHITE, True
            else:
                cell.fill.fore_color.rgb = WHITE if r % 2 else PURPLE_TINT
                colour, bold = INK, (c == 0)
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.CENTER if c else PP_ALIGN.LEFT
            run = para.add_run()
            run.text = str(value)
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = colour
            run.font.name = FONT
    return shape


def figure_slide(prs, title, image, caption, note, kicker="Result"):
    slide = blank(prs)
    fill(slide, WHITE)
    top = heading(slide, title, kicker)
    if image.exists():
        from PIL import Image as PILImage  # noqa: PLC0415
        with PILImage.open(image) as im:
            ratio = im.height / im.width
        max_w, max_h = Inches(8.6), H - top - Inches(1.25)
        width = max_w
        height = Emu(int(width * ratio))
        if height > max_h:
            height = max_h
            width = Emu(int(height / ratio))
        slide.shapes.add_picture(str(image), MARGIN, top, width=width, height=height)
        panel_left = MARGIN + width + Inches(0.35)
        panel_w = W - panel_left - MARGIN
        if panel_w > Inches(1.8):
            panel(slide, panel_left, top, panel_w, height)
            _text(box(slide, panel_left + Inches(0.22), top + Inches(0.22),
                      panel_w - Inches(0.44), height - Inches(0.4)),
                  caption, size=14, color=INK, space_after=10, line=1.3)
    notes(slide, note)
    return slide


# ---------------------------------------------------------------------------
# Live numbers
# ---------------------------------------------------------------------------


def load_metrics() -> dict:
    def read(path):
        p = REPO_ROOT / path
        return json.loads(p.read_text()) if p.exists() else {}

    return {
        "ft_val": read(".local/track_b1_finetune/predictions/metrics_val.json"),
        "ft_test": read(".local/track_b1_finetune/predictions/metrics_test.json"),
        "fz_val": read(".local/track_b1_run_w15/predictions/metrics_val.json"),
        "fz_test": read(".local/track_b1_run_w15/predictions/metrics_test.json"),
        "fz_head": read(".local/track_b1_run_w15/head_results.json"),
    }


def f(metrics, key, path, default="—"):
    node = metrics.get(key) or {}
    for step in path:
        node = (node or {}).get(step) if isinstance(node, dict) else None
    return f"{node:.3f}" if isinstance(node, (int, float)) else default


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------


def build(output: Path, figures: Path) -> None:
    m = load_metrics()
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # --- 1. Title ---------------------------------------------------------
    s = blank(prs); fill(s, PURPLE_DEEP)
    rule(s, Inches(2.35), width=Inches(2.2), color=WHITE)
    _text(box(s, MARGIN, Inches(2.7), Inches(11.2), Inches(2.1)),
          ["Detecting Pickup and Putdown in Retail Video"],
          size=44, color=WHITE, bold=True, space_after=0, line=1.08)
    _text(box(s, MARGIN, Inches(4.35), Inches(10.6), Inches(1.0)),
          ["Actor-conditioned video transformers, and why direction of transfer "
           "does not generalise across recording days"],
          size=20, color=RGBColor(0xD6, 0xC7, 0xEA), space_after=0, line=1.3)
    _text(box(s, MARGIN, Inches(5.9), Inches(10.6), Inches(0.9)),
          ["Track B1  ·  Summer School 2026 Computer Vision Case",
           "42 annotated recordings  ·  259 events  ·  VideoMAE-Base"],
          size=14, color=RGBColor(0xB6, 0xA2, 0xD4), space_after=5)
    notes(s, "Talk is ~20 minutes. Two methodological findings and one negative result. "
             "The negative result is the part worth remembering.")

    # --- 2. The question --------------------------------------------------
    s = blank(prs); fill(s, WHITE)
    top = heading(s, "Did the item leave the shelf, or return to it?", "The question")
    bullets(s, [
        ("Automated checkout needs the direction of transfer, not merely that an "
         "interaction happened.", True),
        "Charging a customer for an item they inspected and put back is worse than "
        "missing the event: it is visible, adversarial, and expensive to dispute.",
        "So per-class performance on the minority class governs deployment — not "
        "aggregate F1.",
    ], top)
    for i, (value, label) in enumerate([
        ("259", "annotated events"), ("170 / 89", "pickup / putdown"),
        ("42", "recordings"), ("5", "recording days")]):
        stat_panel(s, MARGIN + i * Inches(2.95), Inches(5.55), Inches(2.7), value, label)
    notes(s, "Frame the asymmetry early: a false pickup is a charge for something the "
             "shopper returned. That is the error the system must not make.")

    # --- 3. Why it is hard ------------------------------------------------
    s = blank(prs); fill(s, WHITE)
    top = heading(s, "The two classes are near time-reverses", "Why this is hard")
    bullets(s, [
        "Same shelf, same hands, same merchandise, near-identical spatial evidence.",
        ("What separates them is the temporal ordering of a hand-state change and a "
         "shelf-state change.", True),
        "Standard temporal action detection benchmarks reward appearance cues — their "
        "classes differ in object and scene.",
        "Here appearance is shared, so direction is the whole problem.",
    ], top, width=Inches(7.0))
    panel(s, Inches(8.3), top, Inches(4.15), Inches(3.0))
    _text(box(s, Inches(8.6), top + Inches(0.3), Inches(3.6), Inches(2.5)),
          [("Consequence", True),
           "A model can score well on 'an interaction occurred' while learning nothing "
           "about which way the item moved.",
           "That is exactly what we observe on an unseen day."],
          size=14, color=INK, space_after=11, line=1.3)
    notes(s, "This slide sets up the negative result 12 slides later. Direction is the "
             "fragile cue because it is the only one not shared between the classes.")

    # --- 4. Data and annotation -------------------------------------------
    s = blank(prs); fill(s, WHITE)
    top = heading(s, "Annotation directly on source video", "Data")
    bullets(s, [
        "Single fixed overhead camera, 3840 × 2160 at 20 fps, 2–5 minutes per clip.",
        "Anonymised at source before reaching the annotation team.",
        ("Each CVAT track carries a box on every frame it spans — giving both the "
         "temporal extent and the spatial region in one artefact.", True),
        "Seven clips completed with zero events: verified negatives, kept as training "
        "signal rather than discarded.",
        "Frame rate probed per file; the filename-derived estimate drifts 0.35 s by "
        "end of clip, longer than a median event.",
    ], top, width=Inches(7.4))
    panel(s, Inches(8.7), top + Inches(0.1), Inches(3.75), Inches(2.55))
    _text(box(s, Inches(8.95), top + Inches(0.35), Inches(3.3), Inches(2.1)),
          [("Median event", True), "0.90 seconds",
           ("Median annotation box", True), "214 × 249 px in 4K"],
          size=15, color=PURPLE_DEEP, space_after=9, line=1.25)
    notes(s, "The box is the key asset: it removes the need for a pose pipeline, but it "
             "also creates the conditioning caveat raised in the limitations.")

    # --- 5. Pipeline ------------------------------------------------------
    s = blank(prs); fill(s, WHITE)
    top = heading(s, "From annotation to evaluated events", "Pipeline")
    stages = [
        ("CVAT\nexport", "42 archives\n261 intervals"),
        ("Canonical\ntables", "events, clips,\nactor tracks"),
        ("Actor\nwindows", "7,716 labelled\n1.5 s windows"),
        ("Crop\ncache", "decode once\n224 × 224"),
        ("VideoMAE\nclassifier", "3-class\nwindow head"),
        ("Decode +\nevaluate", "smooth, peak,\nsame-type merge"),
    ]
    width, gap = Inches(1.86), Inches(0.16)
    for i, (title_txt, sub) in enumerate(stages):
        left = MARGIN + i * (width + gap)
        panel(s, left, top + Inches(0.4), width, Inches(1.75))
        _text(box(s, left + Inches(0.1), top + Inches(0.6), width - Inches(0.2), Inches(0.8)),
              [title_txt], size=14, color=PURPLE_DEEP, bold=True,
              space_after=0, align=PP_ALIGN.CENTER, line=1.15)
        _text(box(s, left + Inches(0.1), top + Inches(1.38), width - Inches(0.2), Inches(0.7)),
              [sub], size=11, color=MUTED, space_after=0, align=PP_ALIGN.CENTER, line=1.2)
    bullets(s, [
        "Splits assigned by recording day, never by clip — clips minutes apart share "
        "shoppers, lighting and shelf stock.",
        "Build fails if any clip appears under two splits.",
        "Held-out test day read only after the configuration was frozen.",
    ], top + Inches(2.55), size=16)
    notes(s, "One command reproduces this: make track-b1-all.")

    # --- 6. Actor-conditioned windows -------------------------------------
    s = blank(prs); fill(s, WHITE)
    top = heading(s, "One track is one actor stream", "Method")
    bullets(s, [
        "A window is labelled by what occupies its centre, using only that actor's events.",
        ("Labelling per clip instead of per actor stamps one actor's pickup onto a "
         "window cropped around another — 273 overlapping actor pairs in this corpus.", True),
        "Crop fixed per candidate, not per window: a crop that shrinks and grows with "
        "motion correlates with event timing.",
        "Two actors in one clip therefore produce genuinely independent predictions.",
    ], top, width=Inches(7.4))
    panel(s, Inches(8.7), top + Inches(0.1), Inches(3.75), Inches(2.3))
    _text(box(s, Inches(8.95), top + Inches(0.32), Inches(3.3), Inches(1.9)),
          [("Observed in output", True),
           "trk000  putdown  16.35–19.85",
           "trk002  pickup   16.75–19.25",
           "trk001  pickup   17.30–20.30"],
          size=13, color=PURPLE_DEEP, space_after=7, line=1.25)
    notes(s, "Three actors, overlapping in time, different event types — the acceptance "
             "criterion observed rather than argued.")

    # --- 7. Crop leakage --------------------------------------------------
    s = blank(prs); fill(s, WHITE)
    top = heading(s, "Conditioning available only for positives leaks the label",
                  "Design decision 1")
    bullets(s, [
        "Boxes are drawn only while an event is in progress.",
        ("A background window finds no box, falls back to the full 4K frame, and the "
         "classes become separable on crop size alone.", True),
        "The model could reach high accuracy without ever attending to the action — and "
        "nothing in the loss curve would show it.",
        ("Fix: extend each track ±3 s with its boundary box held, so positives and "
         "negatives from a candidate share geometry.", True),
        "The padding is also where the hardest negatives live: approach and withdrawal.",
    ], top)
    notes(s, "Generalises beyond this project: any region-conditioned formulation where "
             "conditioning is available only for positives has this hazard.")

    # --- 8. Compute -------------------------------------------------------
    s = blank(prs); fill(s, WHITE)
    top = heading(s, "Two bottlenecks that decided what was possible", "Engineering")
    table(s, [
        ["", "Naive", "After", "Effect"],
        ["Decode a window (4K, seek per frame)", "20.6 s / 16 frames", "sequential, cached once",
         "~17 h → ~9 min per epoch"],
        ["Frozen backbone recomputed each epoch", "86.2 M params", "embeddings cached once",
         "~9 min → under 1 s per epoch"],
    ], MARGIN, top + Inches(0.15), BODY_W, col_widths=[3.4, 1.9, 2.2, 2.3], size=13)
    bullets(s, [
        "Seeking decodes forward from the preceding keyframe at full resolution; "
        "sequential reads are two orders of magnitude cheaper per frame.",
        ("Cheap epochs are not a convenience — they are what made a 200-combination "
         "threshold sweep and honest early stopping affordable.", True),
        "Embedding caching is valid only while the backbone is frozen.",
    ], top + Inches(1.75), size=16)
    notes(s, "Measured, not estimated: 16 seeking reads 20.6 s against 48 sequential 0.4 s.")

    # --- 9. Gates ---------------------------------------------------------
    s = blank(prs); fill(s, WHITE)
    top = heading(s, "Two gates before any training run", "Verification")
    for i, (name, body) in enumerate([
        ("Gate A — look at the data",
         "Render the frames the loader actually produces, index-stamped, as grids.\n\n"
         "Verified: hand enters, grasps, lifts, item gone. Backgrounds framed like "
         "positives, confirming the padding removed crop leakage."),
        ("Gate B — tiny overfit",
         "Memorise a small batch, on a class-balanced subset.\n\n"
         "Balance matters: the leading manifest rows are one candidate and usually one "
         "class, which a constant predictor memorises — the gate passes while proving "
         "nothing."),
    ]):
        left = MARGIN + i * Inches(6.0)
        panel(s, left, top, Inches(5.6), Inches(3.1))
        _text(box(s, left + Inches(0.3), top + Inches(0.28), Inches(5.0), Inches(0.5)),
              [name], size=18, color=PURPLE_DEEP, bold=True, space_after=0)
        _text(box(s, left + Inches(0.3), top + Inches(0.92), Inches(5.0), Inches(2.0)),
              body.split("\n\n"), size=14, color=INK, space_after=10, line=1.3)
    notes(s, "Both gates caught real problems. Gate A found the crop leakage.")

    # --- 10. FINDING 1 ----------------------------------------------------
    s = blank(prs); fill(s, WHITE)
    top = heading(s, "A window wider than the event caps tIoU",
                  "Finding 1")
    bullets(s, [
        "An interval spanning the first to last contributing window is at least one "
        "window long.",
        ("With a 2.5 s window and a 0.90 s median event, tIoU cannot exceed "
         "0.90 / 2.5 ≈ 0.36 — below the 0.5 operating point, however good the "
         "classifier is.", True),
        "A window is labelled by what sits at its centre, so the centres carry its "
        "temporal evidence.",
        ("Centre-derived boundaries: validation F1 @ tIoU 0.3 from 0.350 to 0.647, "
         "with no retraining at all.", True),
        "Boundary error fell from 1.35 s to 0.21 s.",
    ], top)
    notes(s, "Arithmetic, not a modelling limitation. Worth computing d/w in advance "
             "whenever sliding-window detection is scored with tIoU.")

    figure_slide(prs, "Decoder geometry dominated the ablation",
                 figures / "fig1_decode_ablation.png",
                 [("Same model in all four bars.", True),
                  "Only the decoding of window scores into intervals differs.",
                  "Threshold tuning: +0.086",
                  ("Boundary construction: +0.297", True)],
                 "Emphasise that the model is identical across all four bars.",
                 kicker="Finding 1")

    # --- 12. Model + training --------------------------------------------
    s = blank(prs); fill(s, WHITE)
    top = heading(s, "Frozen probe, then partial fine-tuning", "Model")
    table(s, [
        ["Validation, window level", "Frozen probe", "Fine-tuned (last 2 blocks)"],
        ["macro F1", f(m, "fz_head", ["val_f1_macro"]), "0.743"],
        ["pickup F1", f(m, "fz_head", ["val_f1_per_class", "pickup"]), "0.736"],
        ["putdown F1", f(m, "fz_head", ["val_f1_per_class", "putdown"]), "0.539"],
    ], MARGIN, top + Inches(0.1), Inches(8.0), col_widths=[3.0, 2.0, 2.6], size=14)
    bullets(s, [
        ("Discriminative learning rates are necessary, not optional — head 1e-3, "
         "backbone 5e-5.", True),
        "A uniform 5e-5 left the randomly initialised head predicting background after "
        "two epochs, which first read as evidence that unfreezing fails.",
        "Convergence is non-monotone: epoch 1 reached pickup 0.635 with putdown at "
        "0.057; the optimum arrived at epoch 4.",
        "A short-patience schedule would have produced the opposite conclusion.",
    ], top + Inches(2.0), size=16)
    notes(s, "Honest note: our first fine-tuning run was misconfigured and would have "
             "supported the wrong conclusion.")

    figure_slide(prs, "Validation holds up; the held-out day does not",
                 figures / "fig2_val_test_gap.png",
                 [("Both models, both splits.", True),
                  f"Fine-tuned validation F1 @ 0.3: {f(m, 'ft_val', ['tiou@0.3', 'f1'])}",
                  f"Fine-tuned test F1 @ 0.3: {f(m, 'ft_test', ['tiou@0.3', 'f1'])}",
                  ("The gap is carried almost entirely by putdown.", True)],
                 "Do not rush this slide. The gap, not the headline number, is the result.",
                 kicker="Results")

    # --- 14. FINDING 2 ----------------------------------------------------
    s = blank(prs); fill(s, WHITE)
    top = heading(s, "Direction does not survive a change of recording day", "Finding 2")
    table(s, [
        ["Mean predicted probability on true-putdown windows", "p(putdown)", "p(pickup)"],
        ["Validation day", "0.445", "0.296"],
        ["Held-out test day", "0.062", "0.588"],
    ], MARGIN, top + Inches(0.1), Inches(9.2), col_widths=[4.6, 1.6, 1.6], size=15)
    bullets(s, [
        ("The model is not uncertain about putdowns on the test day — it confidently "
         "calls them pickups.", True),
        "Maximum p(putdown) across all 57 true-putdown test windows is 0.467, so no "
        "threshold recovers the class. This is not a calibration failure.",
        "Detection and temporal localisation transfer; direction does not.",
    ], top + Inches(1.75), size=17)
    notes(s, "The threshold point matters — it is the first objection an audience raises.")

    figure_slide(prs, "No threshold recovers the class",
                 figures / "fig3_putdown_probability_shift.png",
                 [("Predicted putdown probability on windows whose ground truth is "
                   "putdown.", True),
                  "Validation: bimodal, substantial mass above threshold.",
                  ("Test: the entire distribution sits below it.", True)],
                 "This is the single most important slide in the talk.",
                 kicker="Finding 2")

    figure_slide(prs, "Every matched putdown became a pickup",
                 figures / "fig4_confusion.png",
                 [("Rows are ground truth, columns prediction.", True),
                  "Validation: putdowns mostly recovered.",
                  ("Test: all six matched putdowns classified as pickups, none "
                   "correct.", True)],
                 "The confusion matrix makes the failure concrete.",
                 kicker="Finding 2")

    # --- 17. Why it matters ----------------------------------------------
    s = blank(prs); fill(s, WHITE)
    top = heading(s, "What this means for a checkout system", "Implications")
    bullets(s, [
        ("On an unseen day the system behaves as an interaction detector with a "
         "pickup bias.", True),
        "That is the consequential failure mode: a putdown scored as a pickup charges "
        "a customer for an item they returned to the shelf.",
        f"Aggregate F1 of {f(m, 'ft_test', ['tiou@0.3', 'f1'])} does not communicate "
        "this. Per-class recall of 0.000 on putdown does.",
        ("Report the minority class. An aggregate number hid a total class failure.", True),
    ], top)
    panel(s, MARGIN, Inches(5.5), BODY_W, Inches(1.2), fill_color=PURPLE_DEEP, edge=PURPLE_DEEP)
    _text(box(s, MARGIN + Inches(0.4), Inches(5.75), BODY_W - Inches(0.8), Inches(0.8)),
          ["Presence of an interaction is supported by many stable cues. "
           "Direction rests on one fragile cue — and that is the one that fails."],
          size=18, color=WHITE, bold=True, space_after=0, align=PP_ALIGN.CENTER, line=1.25)
    notes(s, "This is the take-home message.")

    # --- 18. Limitations --------------------------------------------------
    s = blank(prs); fill(s, WHITE)
    top = heading(s, "What bounds these claims", "Limitations")
    bullets(s, [
        ("Candidates and crops come from ground-truth annotation boxes, and are used "
         "at inference too. The model is handed the actor region for free; the system "
         "is not deployable as it stands.", True),
        "Only five recording days exist, so validation and test are one day each. Both "
        "estimates are high-variance.",
        "One camera, one store. Nothing here shows the failure is general rather than "
        "specific to this viewpoint.",
        "Actor identity is per-interaction, not per-person.",
        "Verified-negative crops are borrowed from the annotated pool and do not "
        "correspond to a person's location.",
    ], top, size=16)
    notes(s, "State the oracle-conditioning caveat plainly. An audience that spots it "
             "before we mention it will discount everything else.")

    # --- 19. Future work --------------------------------------------------
    s = blank(prs); fill(s, WHITE)
    top = heading(s, "What we would do next", "Future work")
    for i, (num, title_txt, body) in enumerate([
        ("1", "Measure it properly",
         "Day-level grouped cross-validation across all five days. At this scale a "
         "single-day split cannot support the comparisons we want to make."),
        ("2", "Supervise direction explicitly",
         "Time-reversal augmentation, or a frame-order pretext task, so ordering must "
         "be encoded rather than hoped for."),
        ("3", "Fuse an external direction prior",
         "Shelf-state transition sign — object removed against object placed — is "
         "precisely the quantity that fails to transfer."),
    ]):
        left = MARGIN + i * Inches(4.0)
        panel(s, left, top, Inches(3.7), Inches(3.2))
        _text(box(s, left + Inches(0.28), top + Inches(0.24), Inches(1.0), Inches(0.6)),
              [num], size=30, color=PURPLE_MID, bold=True, space_after=0)
        _text(box(s, left + Inches(0.28), top + Inches(0.88), Inches(3.1), Inches(0.6)),
              [title_txt], size=16, color=PURPLE_DEEP, bold=True, space_after=0, line=1.15)
        _text(box(s, left + Inches(0.28), top + Inches(1.6), Inches(3.1), Inches(1.4)),
              [body], size=13, color=INK, space_after=0, line=1.3)
    notes(s, "Ordered by expected value. Cross-validation first — without it no later "
             "comparison can be trusted.")

    # --- 20. Conclusions --------------------------------------------------
    s = blank(prs); fill(s, PURPLE_DEEP)
    rule(s, Inches(1.0), width=Inches(2.2), color=WHITE)
    _text(box(s, MARGIN, Inches(1.35), Inches(11.2), Inches(0.9)),
          ["Conclusions"], size=36, color=WHITE, bold=True, space_after=0)
    _text(box(s, MARGIN, Inches(2.4), Inches(11.2), Inches(3.6)),
          [("The pipeline mattered more than the model.", True),
           "Interval construction moved validation F1 from 0.350 to 0.647 with no "
           "retraining. Held-box padding removed a crop shortcut that would have let the "
           "classifier separate classes without observing the action.",
           ("Detection transfers across days. Direction does not.", True),
           f"Fine-tuning raised validation event F1 to {f(m, 'ft_val', ['tiou@0.3', 'f1'])} "
           f"and test to {f(m, 'ft_test', ['tiou@0.3', 'f1'])}. On the held-out day every "
           "matched putdown was classified as a pickup.",
           ("Direction of transfer needs explicit supervision — it does not emerge from "
            "class labels.", True)],
          size=17, color=RGBColor(0xE4, 0xDA, 0xF2), space_after=14, line=1.3)
    _text(box(s, MARGIN, Inches(6.25), Inches(11.2), Inches(0.6)),
          ["Code, data manifests, weights and evaluation artefacts are released as a "
           "reproducibility bundle."],
          size=13, color=RGBColor(0xB6, 0xA2, 0xD4), space_after=0)
    notes(s, "Close on the negative result. It is the contribution most useful to others.")

    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output))
    print(f"wrote {output}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides, "
          f"{output.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / ".local/paper/track_b1_presentation.pptx")
    parser.add_argument("--figures", type=Path, default=REPO_ROOT / ".local/paper/figures")
    args = parser.parse_args()
    build(args.output, args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
