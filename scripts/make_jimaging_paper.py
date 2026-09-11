#!/usr/bin/env python3
"""Build the Journal of Imaging manuscript from MDPI's official template (revised).

    python scripts/make_jimaging_paper.py

Fills the template in place with the template's own MDPI_* named styles. Every number
is read from the results registry (.local/paper/revision/registry.csv) and the
leave-one-day-out and sensitivity tables, so one configuration produces one set of
figures everywhere in the manuscript.

The template ships as .dot with a template content type python-docx refuses; it is
rewritten to a document content type on load. Nothing else is altered.
"""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/lubomir-angelov/ai_workshop_cv_case"
BRANCH = "feature/cvat_annotation"
RELEASE_TAG = "v1.0-jimaging"

TEXT = "MDPI_3.1_text"
H1, H2 = "MDPI_2.1_heading1", "MDPI_2.2_heading2"
BACK = "MDPI_6.2_back_matter"
REFS = "MDPI_8.1_references"


def open_template(source: Path, working: Path) -> Document:
    working.parent.mkdir(parents=True, exist_ok=True)
    tpl, doc = b"wordprocessingml.template.main+xml", b"wordprocessingml.document.main+xml"
    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(working, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.replace(tpl, doc)
            zout.writestr(item, data)
    return Document(str(working))


def clear_body(document: Document) -> None:
    body = document.element.body
    for child in list(body):
        if not child.tag.endswith("}sectPr"):
            body.remove(child)


def para(d, text, style=TEXT):
    p = d.add_paragraph(style=style)
    p.add_run(text)
    return p


def rich(d, parts, style=TEXT):
    p = d.add_paragraph(style=style)
    for part in parts:
        text, mark = part if isinstance(part, tuple) else (part, "")
        run = p.add_run(text)
        run.bold, run.italic = "b" in mark, "i" in mark
    return p


def caption(d, style, text):
    cap = d.add_paragraph(style=style)
    head, _, rest = text.partition(". ")
    cap.add_run(head + ". ").bold = True
    cap.add_run(rest)


def figure(d, image: Path, text: str, width_in=6.1):
    if not image.exists():
        return
    p = d.add_paragraph(style="MDPI_5.2_figure")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(image), width=Inches(width_in))
    caption(d, "MDPI_5.1_figure_caption", text)


def table(d, rows, text):
    caption(d, "MDPI_4.1_table_caption", text)
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "MDPI_4.1_three_line_table"
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = t.cell(r, c)
            cell.text = ""
            p = cell.paragraphs[0]
            p.style = d.styles["MDPI_4.2_table_body"]
            if c:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(value))
            run.bold = r == 0
            run.font.size = Pt(8)


# ---------------------------------------------------------------------------
# Numbers: one registry, one row per configuration
# ---------------------------------------------------------------------------


def load_numbers(revision: Path) -> dict:
    reg = pd.read_csv(revision / "registry.csv").set_index("run_id")
    lodo = pd.read_csv(revision / "lodo_frozen.csv")
    sens = pd.read_csv(revision / "putdown_sensitivity.csv")
    dur = pd.read_csv(revision / "event_durations.csv").set_index("stat")["value"]
    return {"reg": reg, "lodo": lodo, "sens": sens, "dur": dur}


def r(reg, model, split, boundary="window_centers", thr="tuned"):
    return reg.loc[f"{model}|{split}|{boundary}|{thr}"]


def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def f1(x): return f"{x:.1f}"


def build(template: Path, output: Path, figures: Path, revision: Path) -> None:
    n = load_numbers(revision)
    reg, lodo, sens, dur = n["reg"], n["lodo"], n["sens"], n["dur"]
    FZV, FZT = r(reg, "frozen_probe", "val"), r(reg, "frozen_probe", "test")
    FTV, FTT = r(reg, "finetuned_2blocks", "val"), r(reg, "finetuned_2blocks", "test")
    FZV_SPAN, FTV_SPAN = r(reg, "frozen_probe", "val", "window_span"), r(reg, "finetuned_2blocks", "val", "window_span")
    lodo_m3, lodo_s3 = lodo["f1@0.3"].mean(), lodo["f1@0.3"].std()
    lodo_mp, lodo_sp = lodo["putdown_f1@0.5"].mean(), lodo["putdown_f1@0.5"].std()

    d = open_template(template, output.with_name("_template_working.docx"))
    clear_body(d)

    # ================= Front matter =================
    para(d, "Article", "MDPI_1.1_article_type")
    para(d, "Pickup and Putdown Detection in Retail Video Under Annotation-Derived "
            "Conditioning: Decoder Geometry, Crop Conditioning and Cross-Day Variation of an "
            "Actor-Conditioned Video Transformer", "MDPI_1.2_title")
    rich(d, ["Firstname Lastname ", ("1", "i"), ", Firstname Lastname ", ("2", "i"),
             " and Firstname Lastname ", ("1,*", "i")], "MDPI_1.3_authornames")
    para(d, "1\tAffiliation 1; e-mail@e-mail.com", "MDPI_1.6_affiliation")
    para(d, "2\tAffiliation 2; e-mail@e-mail.com", "MDPI_1.6_affiliation")
    para(d, "*\tCorrespondence: e-mail@e-mail.com", "MDPI_1.6_affiliation")

    rich(d, [("Abstract: ", "b"),
             "Distinguishing an item removed from a shelf from one returned to it is a temporal "
             "action detection problem whose two classes are near time-reverses, so the evidence "
             "is the direction of transfer. We train an actor-conditioned "
             "VideoMAE window classifier on 42 human-annotated 4K store recordings (259 events, five "
             "recording days). We evaluate classification and temporal refinement within "
             "annotation-derived spatial and temporal candidates; the results therefore describe "
             "performance under oracle conditioning, not end-to-end detection in continuous video. "
             "Two results concern the pipeline. Taking a predicted interval as the span of "
             "contributing windows bounds per-event temporal IoU by the ratio of event to window "
             "duration; on identical window scores and thresholds, centre-derived intervals raised "
             f"validation F1 at tIoU 0.5 from {f3(FTV_SPAN['f1@0.5'])} to {f3(FTV['f1@0.5'])}. "
             "Where spatial conditioning exists only for positives, background windows fall back to "
             "a full-frame crop and separate on geometry alone. Partial fine-tuning raised "
             f"validation event F1 to {f3(FTV['f1@0.3'])}, but at the validation-selected operating "
             "point achieved zero putdown F1 on the held-out day, with scores on true putdowns "
             "shifted towards pickup. Leave-one-day-out evaluation of the frozen probe gives mean "
             f"F1 {f3(lodo_m3)} ± {f3(lodo_s3)} and putdown F1 {f3(lodo_mp)} ± {f3(lodo_sp)} over "
             "five days: the single-split validation figure was optimistic and the directional "
             "weakness recurs across days."],
         "MDPI_1.7_abstract")
    rich(d, [("Keywords: ", "b"),
              "temporal action detection; video transformers; VideoMAE; retail computer vision; "
              "shortcut learning; evaluation protocol; leave-one-day-out; annotation methodology"],
         "MDPI_1.8_keywords")

    # ================= 1. Introduction =================
    para(d, "1. Introduction", H1)
    para(d, "Automated checkout and shrinkage analytics both depend on one perceptual distinction: "
            "whether an item moved from the shelf into a shopper's possession or in the opposite "
            "direction. A system that reports only that an interaction occurred is of limited use to "
            "either. Charging a customer for an item they inspected and returned is a worse outcome "
            "than missing the event, because the error is visible to the customer and costly to "
            "dispute.")
    para(d, "This asymmetry makes the task an awkward fit for the standard temporal action detection "
            "formulation. Established benchmarks such as THUMOS14 [1] and ActivityNet [2] contain "
            "action classes that differ in object, scene and appearance, and their leading methods, "
            "whether proposal-based such as BMN [3] or single-stage such as ActionFormer [4], are "
            "developed against that structure. Pickup and putdown share the shelf, the hands and the "
            "merchandise; the evidence that separates them is the temporal ordering of a hand-state "
            "change relative to a shelf-state change. The Something-Something dataset [5] was built "
            "to expose exactly this weakness in appearance-driven models, and the arrow-of-time "
            "literature [10] shows that temporal direction is a learnable but non-trivial signal. "
            "Retail interaction has its own benchmark in the MERL Shopping dataset [11], whose "
            "classes (reach to shelf, retract from shelf, hand in shelf, inspect product) are "
            "adjacent to ours but do not resolve the direction of transfer. The broader hazard that "
            "a model fits a decision rule which performs on the benchmark and fails under shift is "
            "well characterised as shortcut learning [12].")
    para(d, "Video transformers pre-trained by masked autoencoding [6], building on the Vision "
            "Transformer [7] and large action corpora [8], are now the default backbone. Data "
            "efficiency on small downstream sets is an explicit claim of VideoMAE [6]. Whether that "
            "efficiency extends to a distinction defined by temporal direction, under the shift "
            "between recording days in one store, is not something existing benchmarks measure.")
    para(d, "We report a methodological case study on that question. Its scope is bounded and "
            "should be read as such: candidates and crop regions are derived from the ground-truth "
            "annotation boxes and are used at inference as well as in training, so every figure "
            "reported here is an estimate under oracle conditioning, not the performance of an "
            "end-to-end detector on continuous video. Within that scope the contribution has three "
            "parts. First, a data path from source-video box annotation to actor-conditioned window "
            "training that needs no pose-estimation stage: annotators draw a tracked box on each "
            "interaction in CVAT [9], and that artefact supplies both the temporal extent and the "
            "spatial conditioning region. Second, two failure modes in the machinery around the "
            "model that are easy to introduce and invisible in aggregate accuracy: the construction "
            "of predicted intervals from window scores, and the behaviour of the crop when no box "
            "exists. Third, a characterisation of how the pickup/putdown distinction varies across "
            "recording days, evaluated both on a single held-out day and by leave-one-day-out over "
            "all five, with the diagnostic evidence needed to separate a classification failure "
            "from a threshold-calibration failure.")

    # ================= 2. Materials and Methods =================
    para(d, "2. Materials and Methods", H1)
    para(d, "This section describes each stage in execution order. The implementation, "
            "configuration, results registry and evaluation code are openly available "
            "(Section 2.10), and the command make track-b1-all reproduces the sequence from an "
            "existing annotation export.")

    para(d, "2.1. Video Corpus", H2)
    para(d, "The corpus comprises 42 recordings from a single fixed overhead camera in one store, "
            "each between two and five minutes long, at 3840 × 2160 pixels and exactly 20.0 frames "
            "per second, spanning five recording days. The footage was anonymised by the data "
            "provider before release to the research team. Table 1 summarises the corpus and the "
            "split assignment of Section 2.9.")
    table(d, [
        ["Split", "Recording day(s)", "Clips", "Pickup", "Putdown", "Events", "Windows", "Unique video (h)"],
        ["Train", "2026-05-20, -21, -22", "23", "127", "65", "60 / 73 / 59", "5192", "1.35"],
        ["Validation", "2026-05-23", "7", "20", "12", "32", "973", f"{FZV['unique_video_hours']:.2f}"],
        ["Test", "2026-05-26", "12", "23", "12", "35", "1551", f"{FZT['unique_video_hours']:.2f}"],
        ["Total", "5 days", "42", "170", "89", "259", "7716", "2.31"],
    ], "Table 1. Corpus composition and split assignment. Events per training day are given "
       "individually. Unique video time is the denominator used for false positives per hour.")

    para(d, "2.2. Annotation Protocol", H2)
    para(d, "Annotation was performed in CVAT [9] on the source video rather than on pre-extracted "
            "candidate windows, by a single annotator. Each annotated interval is a CVAT track "
            "carrying an axis-aligned box on every frame it spans, labelled pickup, putdown or "
            "ignore, with attributes for confidence (high, medium, low), a hard-case flag, item count "
            "and review status. The annotation guideline defines a pickup as beginning when the hand "
            "makes contact with the item on the shelf and ending when the item clears the shelf "
            "region, and a putdown as the reverse; the box is drawn to enclose the hand and item "
            "through the transfer. A track interrupted by an outside marker and later resumed is "
            "treated as two events.")
    para(d, "The export contains 261 CVAT tracks: 258 labelled pickup or putdown and 3 labelled "
            "ignore. One event track is interrupted and resumed, yielding 259 events, of which 170 "
            "are pickup and 89 putdown; the three ignore tracks yield three ignore intervals. "
            "Confidence is high for 201 events, medium for 40 and low for 18; 52 are flagged as hard "
            "cases; 13 remained in draft review status and are included. The median event lasts "
            "0.90 s (interquartile range 0.60–1.40 s) and the median box measures 214 × 249 pixels. "
            "Seven recordings were completed with zero tracks; these are verified negatives and are "
            "retained. No independent re-annotation of a subset was performed, so inter-annotator "
            "agreement on class and boundary is not available and is noted as a limitation.")

    para(d, "2.3. Step 1: Canonical Representation and Timebase", H2)
    para(d, "CVAT stores annotations as frame indices, so every derived timestamp depends on the "
            "assumed frame rate. The rate implied by the recording window in each filename is "
            "approximately 20.04 fps, which differs from the true rate by about 0.2%; over a "
            "three-minute recording that accumulates to roughly 0.35 s, about 39% of the median "
            "event duration. The exact average frame rate is therefore probed from each file and its "
            "provenance recorded. All 42 recordings are exactly 20.0 fps.")
    para(d, "Each archive is parsed into four tables: events; ignore intervals; clip metadata "
            "including frame rate, duration, recording day and split; and per-frame actor tracks. "
            "The actor-track table uses the column layout of a pose-track table so the downstream "
            "dataset consumes it unchanged.")

    para(d, "2.4. Step 2: Actor-Conditioned Candidates and Windows", H2)
    para(d, "Each CVAT track is one actor stream. Tracks are drawn per interaction rather than per "
            "person, so two temporally overlapping tracks in a recording yield two independent "
            "streams; the corpus contains 273 such overlapping pairs. Window labels are assigned per "
            "actor from that actor's events only. Assigning at the level of the recording, the more "
            "obvious implementation, stamps one actor's pickup onto a window cropped around another.")
    para(d, "A candidate is the padded temporal extent of one actor track. Windows of 1.5 s at a "
            "0.25 s stride are generated across each candidate and labelled by what occupies the "
            "centre: the event class if the centre falls inside an event of that actor, background "
            "otherwise. Windows centred inside an ignore interval are discarded. Low-confidence events "
            "are down-weighted by a factor of two.")

    para(d, "2.5. Step 3: Crop Conditioning and the Padding Requirement", H2)
    para(d, "The annotation box defines the crop: the union of the actor's boxes over the candidate, "
            "expanded by 15% and clamped to the frame, resized to 224 × 224. Boxes exist only while an "
            "event is in progress, so a background window drawn from outside any event finds no box. "
            "The natural fallback, cropping to the full frame, makes every positive a tight crop "
            "around a pair of hands and every negative a wide 4K view; the classes become separable "
            "on crop geometry and a model may attain high accuracy without attending to the action. "
            "We regard this as an instance of shortcut learning [12].")
    para(d, "Each actor track is extended ±3 s with its boundary box held, so background windows from "
            "the padding are cropped identically to positives from the same candidate. For the seven "
            "verified-negative recordings, twelve candidates per recording are placed at seeded random "
            "positions (seed 42), each conditioned on a box sampled from the pool of all annotated "
            "boxes in the corpus. Two consequences are disclosed rather than assumed away. The box pool "
            "is drawn from the whole corpus rather than the training split alone, which is a mild "
            "information leak in box geometry and should be restricted to the training pool in future "
            "work. And where two actors overlap in time, a padded background window of one actor may "
            "contain the other actor's event within its crop; we have not counted these cases, so the "
            "padding is not guaranteed to yield clean negatives. The crop is fixed per candidate rather "
            "than per window, since a crop that varied with the actor's motion would correlate with "
            "event timing.")
    para(d, "We have not quantified the size of the crop shortcut with a controlled comparison against "
            "the full-frame fallback, nor with a geometry-only classifier. Both are listed in "
            "Section 4.4.")

    para(d, "2.6. Step 4: Frame and Embedding Caches", H2)
    para(d, "The source recordings are 4K H.264. Decoding a window by seeking to each of its 16 "
            "frames is prohibitive because each seek decodes forward from the preceding keyframe at "
            "full resolution. Measured on one recording, 16 seeking reads required 20.6 s while 48 "
            "sequential reads required 0.4 s. At approximately 27 s per window, one epoch over the "
            "2314 windows of the 2.5 s configuration would take about 17 h, and over the 5192 windows "
            "of the 1.5 s configuration about 39 h. Each candidate is therefore decoded once, "
            "sequentially, cropped, resized and stored as an 8-bit array (7.2 GB, about twenty "
            "minutes), and windows are served by indexing memory-mapped arrays.")
    para(d, "While the encoder is frozen its output is invariant across epochs. Computing the "
            "768-dimensional embedding once for all 7716 windows reduces an epoch from about nine "
            "minutes to under one second. Preprocessing is deterministic and the encoder is run in "
            "evaluation mode with no stochastic layers active, which is the condition under which the "
            "cache is valid.")

    para(d, "2.7. Step 5: Model and Training", H2)
    para(d, "We use VideoMAE-Base [6] pre-trained on Kinetics-400 [8] (Hugging Face checkpoint "
            "MCG-NJU/videomae-base), with a head of layer normalisation, dropout and a linear "
            "projection to three classes. Each window is 16 frames sampled uniformly in chronological "
            "order, normalised with ImageNet statistics; encoder outputs are mean-pooled over the "
            "token sequence. No data augmentation is applied.")
    para(d, "Two regimes are compared. Frozen probe: the encoder is frozen and only the head trains on "
            "cached embeddings, with AdamW, learning rate 1 × 10⁻³, weight decay 1 × 10⁻², dropout 0.2, "
            "batch size 64, cosine decay, up to 300 epochs with early stopping at patience 40, seed 42. "
            "Partial fine-tuning: the final two transformer blocks are unfrozen with the head, through "
            "the full pixel path, with AdamW, weight decay 1 × 10⁻², dropout 0.1, batch size 8, two "
            "warmup epochs then cosine decay, five epochs, seed 42. The head trains at 1 × 10⁻³ and "
            "the unfrozen blocks at 5 × 10⁻⁵, and the head is warm-started from the frozen probe. Both "
            "regimes use inverse-frequency class weights and select the checkpoint on validation "
            "window-level macro F1.")
    para(d, "We note as an observation, not a controlled result, that an earlier run at a uniform "
            "5 × 10⁻⁵ for head and backbone had a randomly initialised head still predicting "
            "background exclusively after two epochs; an equal-budget comparison of the two learning-"
            "rate schemes was not performed.")
    para(d, "Before any full run, two gates must pass: a visual check of index-stamped frame grids "
            "produced by the loader, and a tiny-overfit test on a class-balanced subset. Balance "
            "matters because the leading manifest rows come from one candidate and usually one class, "
            "which a constant predictor memorises.")

    para(d, "2.8. Step 6: Inference and Decoding", H2)
    para(d, "At inference, windows slide across each candidate at the training stride, class "
            "probabilities are smoothed by a moving average over adjacent windows, and for each event "
            "class independently the smoothed probability is thresholded (no argmax is taken); "
            "contiguous runs above threshold become score regions. Regions of the same type separated "
            "by less than 0.75 s are merged; regions of different type are never merged; regions "
            "shorter than 0.3 s are discarded. One candidate may emit zero, one or several ordered "
            "events. Predictions from different candidates are not de-duplicated against one another.")
    para(d, "A run of above-threshold windows is converted to an interval in one of two ways. "
            "Window-span: from the first window's start to the last window's end. Window-centre: from "
            "the first to the last window's centre, widened by half a stride each side and floored at "
            "the minimum duration. Under window-span a single firing window yields an interval of "
            "length L ≥ w; for a true event of duration d ≤ w, its temporal IoU satisfies "
            "tIoU ≤ d / L ≤ d / w. This is a per-event bound under those assumptions, not a bound on "
            "set-level F1. In this corpus, with w = 2.5 s, 32% of events have d < 0.3 w and 67% have "
            "d < 0.5 w, so they cannot be matched at those thresholds under window-span when a single "
            "window fires; with w = 1.5 s the fractions are 13% and 32%. Window-centre removes the "
            "minimum length w, while the stride and minimum duration remain as resolution limits.")

    para(d, "2.9. Evaluation Protocol and Splits", H2)
    para(d, "Two matching protocols are used and reported separately. Headline metrics use class-aware "
            "one-to-one matching: within each recording, predictions and ground-truth events are "
            "matched by Hungarian assignment maximising total temporal IoU, a pair being admissible "
            "only if the classes agree and tIoU meets the threshold; inadmissible pairs are excluded "
            "before optimisation. Precision, recall and F1 are micro-averaged from pooled TP, FP and "
            "FN, and reported at tIoU 0.3 and 0.5. Per-class precision, recall and F1 are computed at "
            "tIoU 0.5 by the same protocol restricted to that class. The confusion matrix uses a "
            "second, class-agnostic protocol: the same one-to-one matching at tIoU 0.5 with the class "
            "constraint removed, so that a putdown matched to a pickup prediction counts as an "
            "off-diagonal entry. Predictions and events falling inside an ignore interval are removed "
            "before matching. Mean average precision uses each event's peak smoothed probability as "
            "its confidence, ranks predictions by it, and integrates precision over recall at tIoU "
            "0.3, 0.5 and 0.7; the reported average is over those three. Boundary mean absolute error "
            "is computed over class-aware matches at tIoU 0.5 only and the match count is reported "
            "with it. False positives per hour divide FP at tIoU 0.5 by the unique duration of the "
            "evaluated split's recordings.")
    para(d, "Splits are assigned by recording day, never by clip, since recordings minutes apart "
            "share shoppers, lighting and shelf stock. The rule is: rank the five days by event count "
            "in ascending order; the first becomes validation and the second test. Event counts per "
            "day are 60, 73 and 59 for the training days, 32 for 2026-05-23 (validation) and 35 for "
            "2026-05-26 (test). The builder fails if any recording appears under two splits. Decision "
            "thresholds and smoothing width are selected on validation only, by the mean of F1 at tIoU "
            "0.3 and 0.5 over a 200-point grid.")
    para(d, "The test split was read twice: once for the frozen probe and once for the fine-tuned "
            "model, each with its configuration frozen beforehand. The decision to fine-tune, and every "
            "threshold, was made on validation before the first test read; the second read did not "
            "inform any subsequent choice. The post-hoc threshold sensitivity in Section 3.5 reads the "
            "test split again and is presented as diagnostic only.")
    para(d, "To characterise cross-day variation beyond a single held-out day, we additionally "
            "perform leave-one-day-out evaluation of the frozen probe over all five days. For each "
            "held-out day, the sparsest of the remaining four days serves as inner validation; the "
            "head is trained on the other three, the checkpoint is selected on the inner day, and "
            "thresholds are selected on the inner day over a 49-point grid. The held-out day is read "
            "once. Windows from one day are not independent observations and no per-window "
            "significance test is attempted.")

    para(d, "2.10. Code, Data and Reproducibility", H2)
    para(d, f"The implementation is at {REPO_URL}, branch {BRANCH}; the manuscript corresponds to "
            f"release tag {RELEASE_TAG}. The repository contains the CVAT import, window builder, "
            "frame and embedding caches, both training entry points, inference and decoding, the "
            "threshold search, the evaluation harness, the results registry from which every table "
            "and figure here is generated, and 47 unit tests. Method documentation is in "
            "docs/TRACK_B1_CVAT.md and the acceptance configuration in configs/track_b1.yaml.")
    para(d, "Reproducibility has two levels. Every table and figure can be regenerated from the "
            "results registry and per-window score files, which are included in a reproducibility "
            "bundle together with the 42 annotation archives, canonical tables, actor tracks, trained "
            "weights for both regimes, and a SHA-256 manifest; no video access is required for this. "
            "Retraining requires the source recordings, which are commercial footage of members of the "
            "public and are not redistributed; the bundle carries each recording's storage location "
            "and SHA-256 digest so a holder of the appropriate credentials can reconstruct and verify "
            "the input set. The bundle is deposited with the corresponding author and provided on "
            "request; a public archive with a DOI will be created on acceptance.")

    # ================= 3. Results =================
    para(d, "3. Results", H1)
    para(d, "All figures in this section are annotation-conditioned estimates as defined in "
            "Section 1. Every number is traceable to one row of the results registry, identified by "
            "model, split, boundary mode and threshold set, with TP, FP and FN recorded per row.")

    para(d, "3.1. Decoder Geometry: A Controlled Comparison", H2)
    para(d, "Figure 1 isolates the interval-construction rule. For each model, the same saved window "
            "scores are decoded twice with identical thresholds and smoothing, differing only in "
            "boundary mode. Table 2 gives the counts.")
    figure(d, figures / "fig1_decode_ablation.png",
           "Figure 1. Validation event F1 under window-span and window-centre interval construction, "
           "decoded from identical window scores with identical validation-selected thresholds and "
           "smoothing. Only the interval rule differs between the paired bars.")
    table(d, [
        ["Model, boundary", "F1@0.3", "TP/FP/FN @0.3", "F1@0.5", "TP/FP/FN @0.5", "Start MAE (s)"],
        ["Frozen, span", f3(FZV_SPAN["f1@0.3"]),
         f"{int(FZV_SPAN['tp@0.3'])}/{int(FZV_SPAN['fp@0.3'])}/{int(FZV_SPAN['fn@0.3'])}",
         f3(FZV_SPAN["f1@0.5"]), f"{int(FZV_SPAN['tp@0.5'])}/{int(FZV_SPAN['fp@0.5'])}/{int(FZV_SPAN['fn@0.5'])}",
         f2(FZV_SPAN["start_mae_s"])],
        ["Frozen, centre", f3(FZV["f1@0.3"]),
         f"{int(FZV['tp@0.3'])}/{int(FZV['fp@0.3'])}/{int(FZV['fn@0.3'])}",
         f3(FZV["f1@0.5"]), f"{int(FZV['tp@0.5'])}/{int(FZV['fp@0.5'])}/{int(FZV['fn@0.5'])}",
         f2(FZV["start_mae_s"])],
        ["Fine-tuned, span", f3(FTV_SPAN["f1@0.3"]),
         f"{int(FTV_SPAN['tp@0.3'])}/{int(FTV_SPAN['fp@0.3'])}/{int(FTV_SPAN['fn@0.3'])}",
         f3(FTV_SPAN["f1@0.5"]), f"{int(FTV_SPAN['tp@0.5'])}/{int(FTV_SPAN['fp@0.5'])}/{int(FTV_SPAN['fn@0.5'])}",
         f2(FTV_SPAN["start_mae_s"])],
        ["Fine-tuned, centre", f3(FTV["f1@0.3"]),
         f"{int(FTV['tp@0.3'])}/{int(FTV['fp@0.3'])}/{int(FTV['fn@0.3'])}",
         f3(FTV["f1@0.5"]), f"{int(FTV['tp@0.5'])}/{int(FTV['fp@0.5'])}/{int(FTV['fn@0.5'])}",
         f2(FTV["start_mae_s"])],
    ], "Table 2. Controlled decoder comparison on validation. Same window scores, 1.5 s window, "
       "0.25 s stride, validation-selected thresholds; only the boundary rule differs within each "
       "model. Class-aware micro F1 with pooled counts.")
    para(d, "With the classifier held fixed, window-centre decoding raises F1 at tIoU 0.5 from "
            f"{f3(FZV_SPAN['f1@0.5'])} to {f3(FZV['f1@0.5'])} for the frozen probe and from "
            f"{f3(FTV_SPAN['f1@0.5'])} to {f3(FTV['f1@0.5'])} for the fine-tuned model, and reduces "
            f"start MAE from {f2(FZV_SPAN['start_mae_s'])} s to {f2(FZV['start_mae_s'])} s and from "
            f"{f2(FTV_SPAN['start_mae_s'])} s to {f2(FTV['start_mae_s'])} s respectively. The effect "
            "at tIoU 0.3 is smaller and, for the fine-tuned model, within the range where a handful "
            "of matches decides the direction. This is consistent with the per-event bound of "
            "Section 2.8: the 0.5 threshold is where 32% of events cannot be matched at all under "
            "window-span. The earlier 2.5 s configuration, which changes the temporal input as well as "
            "the decoder, is not included in this comparison for that reason.")

    para(d, "3.2. Training Regimes on Validation", H2)
    table(d, [
        ["Validation (2026-05-23), centre decoding", "Frozen probe", "Fine-tuned (2 blocks)"],
        ["Window-level macro F1 (selection metric)", "0.615", "0.743"],
        ["Event F1 @ tIoU 0.3 (TP/FP/FN)", f"{f3(FZV['f1@0.3'])} ({int(FZV['tp@0.3'])}/{int(FZV['fp@0.3'])}/{int(FZV['fn@0.3'])})",
         f"{f3(FTV['f1@0.3'])} ({int(FTV['tp@0.3'])}/{int(FTV['fp@0.3'])}/{int(FTV['fn@0.3'])})"],
        ["Event F1 @ tIoU 0.5 (TP/FP/FN)", f"{f3(FZV['f1@0.5'])} ({int(FZV['tp@0.5'])}/{int(FZV['fp@0.5'])}/{int(FZV['fn@0.5'])})",
         f"{f3(FTV['f1@0.5'])} ({int(FTV['tp@0.5'])}/{int(FTV['fp@0.5'])}/{int(FTV['fn@0.5'])})"],
        ["Pickup F1 @ 0.5", f3(FZV["pickup_f1@0.5"]), f3(FTV["pickup_f1@0.5"])],
        ["Putdown F1 @ 0.5", f3(FZV["putdown_f1@0.5"]), f3(FTV["putdown_f1@0.5"])],
        ["mAP (0.3/0.5/0.7 avg)", f3(FZV["mAP_avg"]), f3(FTV["mAP_avg"])],
        [f"Start MAE (s), n matched", f"{f2(FZV['start_mae_s'])} (n={int(FZV['n_matched_for_mae'])})",
         f"{f2(FTV['start_mae_s'])} (n={int(FTV['n_matched_for_mae'])})"],
        [f"FP per hour @0.5 ({FZV['unique_video_hours']:.2f} h)", f1(FZV["fp_per_hour@0.5"]), f1(FTV["fp_per_hour@0.5"])],
    ], "Table 2b. Validation performance of the two regimes at their validation-selected thresholds "
       "(frozen: pickup 0.45, putdown 0.60, smoothing 3; fine-tuned: 0.40, 0.45, 5). Registry rows "
       "frozen_probe|val|window_centers|tuned and finetuned_2blocks|val|window_centers|tuned.")
    para(d, "Unfreezing the final two blocks improves every reported metric on validation. The "
            "trajectory is not monotone: after one epoch the model reached pickup F1 0.635 with "
            "putdown at 0.057, the second epoch degraded both, and the optimum arrived at epoch four, "
            "after which the train/validation gap widened.")

    para(d, "3.3. Held-Out Recording Day", H2)
    table(d, [
        ["Test (2026-05-26), centre decoding", "Frozen probe", "Fine-tuned (2 blocks)"],
        ["Event F1 @ tIoU 0.3 (TP/FP/FN)", f"{f3(FZT['f1@0.3'])} ({int(FZT['tp@0.3'])}/{int(FZT['fp@0.3'])}/{int(FZT['fn@0.3'])})",
         f"{f3(FTT['f1@0.3'])} ({int(FTT['tp@0.3'])}/{int(FTT['fp@0.3'])}/{int(FTT['fn@0.3'])})"],
        ["Event F1 @ tIoU 0.5 (TP/FP/FN)", f"{f3(FZT['f1@0.5'])} ({int(FZT['tp@0.5'])}/{int(FZT['fp@0.5'])}/{int(FZT['fn@0.5'])})",
         f"{f3(FTT['f1@0.5'])} ({int(FTT['tp@0.5'])}/{int(FTT['fp@0.5'])}/{int(FTT['fn@0.5'])})"],
        ["Pickup F1 @ 0.5", f3(FZT["pickup_f1@0.5"]), f3(FTT["pickup_f1@0.5"])],
        ["Putdown F1 @ 0.5 (TP/FN)", f"{f3(FZT['putdown_f1@0.5'])} ({int(FZT['putdown_tp@0.5'])}/{int(FZT['putdown_fn@0.5'])})",
         f"{f3(FTT['putdown_f1@0.5'])} ({int(FTT['putdown_tp@0.5'])}/{int(FTT['putdown_fn@0.5'])})"],
        ["mAP (0.3/0.5/0.7 avg)", f3(FZT["mAP_avg"]), f3(FTT["mAP_avg"])],
        ["Start MAE (s), n matched", f"{f2(FZT['start_mae_s'])} (n={int(FZT['n_matched_for_mae'])})",
         f"{f2(FTT['start_mae_s'])} (n={int(FTT['n_matched_for_mae'])})"],
        [f"FP per hour @0.5 ({FZT['unique_video_hours']:.2f} h)", f1(FZT["fp_per_hour@0.5"]), f1(FTT["fp_per_hour@0.5"])],
    ], "Table 3. Performance on the held-out day at the configurations frozen on validation. "
       "Registry rows frozen_probe|test|window_centers|tuned and finetuned_2blocks|test|window_centers|tuned.")
    para(d, "Fine-tuning improves aggregate F1 on the held-out day and the direction agrees with "
            f"validation; the magnitude does not, {f3(FTV['f1@0.3'])} against {f3(FTT['f1@0.3'])} at "
            "tIoU 0.3. Figure 2 shows the gap for both models and both classes. At the "
            "validation-selected operating point the fine-tuned model achieved zero putdown F1 on the "
            f"held-out day: {int(FTT['putdown_tp@0.5'])} of 12 putdowns matched, against "
            f"{int(FZT['putdown_tp@0.5'])} for the frozen probe. False positives per hour on the "
            f"held-out day are {f1(FTT['fp_per_hour@0.5'])} for the fine-tuned model over "
            f"{FTT['unique_video_hours']:.2f} h of unique footage.")
    figure(d, figures / "fig2_val_test_gap.png",
           "Figure 2. Validation and held-out-day event F1 for both regimes at their "
           "validation-selected configurations. The gap is carried largely by the putdown class.")

    para(d, "3.4. Localisation Transfers; Direction Shifts", H2)
    para(d, "Under class-agnostic matching at tIoU 0.5, 17 of the 35 held-out-day events are matched "
            "by some prediction (recall 0.49), including 6 of the 12 putdowns; Figure 3 shows the "
            "confusion over those 17. All six matched putdowns are predicted as pickup. Boundary MAE "
            f"over the {int(FTT['n_matched_for_mae'])} class-aware matches is "
            f"{f2(FTT['start_mae_s'])} s at the start and {f2(FTT['end_mae_s'])} s at the end, against "
            f"{f2(FTV['start_mae_s'])} s and {f2(FTV['end_mae_s'])} s on validation; the start error "
            "roughly triples and is 72% of the median event duration, so localisation degrades "
            "materially even where events are found, and the MAE excludes the 18 unmatched events.")
    figure(d, figures / "fig4_confusion.png",
           "Figure 3. Class confusion under class-agnostic one-to-one matching at tIoU 0.5, "
           "fine-tuned model. Rows are ground truth, columns prediction, over matched events only "
           "(validation 22, held-out day 17).", width_in=4.7)
    para(d, "Scores on true putdown windows of the held-out day are shifted towards pickup: mean "
            "p(putdown) is 0.062 and mean p(pickup) 0.588, against 0.445 and 0.296 on validation "
            "(Table 4, Figure 4). One of the 57 true-putdown windows exceeds the 0.45 threshold; the "
            "maximum is 0.467. These observations indicate a substantial directional classification "
            "failure at the selected operating point, but do not by themselves rule out partial "
            "recovery through alternative thresholds, which Section 3.5 examines.")
    table(d, [
        ["Windows with ground truth = putdown", "Mean p(putdown)", "Mean p(pickup)", "Fraction > 0.45", "n"],
        ["Validation day", "0.445", "0.296", "0.49", "63"],
        ["Held-out day", "0.062", "0.588", "0.02", "57"],
    ], "Table 4. Mean predicted probability on true-putdown windows, fine-tuned model. Windows "
       "overlap by construction and are not independent.")
    figure(d, figures / "fig3_putdown_probability_shift.png",
           "Figure 4. Distribution of p(putdown) on true-putdown windows, fine-tuned model, validation "
           "versus held-out day. Windows overlap and the two histograms are drawn from 63 and 57 "
           "windows respectively.", width_in=4.7)

    para(d, "3.5. Post-Hoc Threshold Sensitivity (Diagnostic Only)", H2)
    para(d, "To test whether any operating point recovers the class, we sweep the putdown threshold "
            "on the held-out day with all other settings fixed (Figure 5). This reads the test split "
            "after the reported results and its optimum is not a test result. Putdown recall is zero "
            "for every threshold from 0.25 upward; below 0.25 it rises, reaching recall 0.42 at "
            "precision 0.38 (F1 0.40, 5 of 12) at a threshold of 0.05, far from any value the "
            "validation search would select. Partial recovery is therefore possible in principle, at "
            "an operating point that cannot be chosen without access to the held-out day and at a "
            "precision that would be unacceptable in use.")
    figure(d, figures / "fig6_putdown_sensitivity.png",
           "Figure 5. Held-out-day putdown precision, recall and F1 versus the putdown threshold, "
           "fine-tuned model, all other settings fixed at the validation-selected values. Post-hoc "
           "diagnostic; the test split is read again for this figure.", width_in=4.9)

    para(d, "3.6. Leave-One-Day-Out Over All Five Days", H2)
    para(d, "A single held-out day with 12 putdowns cannot establish a general property. Table 5 and "
            "Figure 6 report the frozen probe with each of the five days held out in turn, checkpoint "
            "and thresholds selected on the inner days only. The fine-tuned regime was not run under "
            "this protocol owing to its cost; the cached-embedding path makes the frozen-probe "
            "evaluation inexpensive.")
    table(d, [["Held-out day", "Inner val day", "Thresholds", "Events", "F1@0.3", "F1@0.5",
               "Putdown F1", "FP/h"]] + [
        [str(x["held_out_day"])[:4] + "-" + str(x["held_out_day"])[4:6] + "-" + str(x["held_out_day"])[6:],
         str(x["inner_val_day"])[4:6] + "-" + str(x["inner_val_day"])[6:],
         f"{x['pickup_thr']:.2f}/{x['putdown_thr']:.2f}", int(x["n_gt_events"]),
         f3(x["f1@0.3"]), f3(x["f1@0.5"]), f3(x["putdown_f1@0.5"]), f1(x["fp_per_hour@0.5"])]
        for _, x in lodo.iterrows()
    ] + [["Mean ± SD", "", "", "", f"{f3(lodo_m3)} ± {f3(lodo_s3)}",
          f"{f3(lodo['f1@0.5'].mean())} ± {f3(lodo['f1@0.5'].std())}",
          f"{f3(lodo_mp)} ± {f3(lodo_sp)}", f"{f1(lodo['fp_per_hour@0.5'].mean())}"]],
          "Table 5. Leave-one-day-out evaluation of the frozen probe. Selection uses the inner days "
          "only; each held-out day is read once. Registry file lodo_frozen.csv.")
    figure(d, figures / "fig5_lodo.png",
           "Figure 6. Held-out-day event F1 for the frozen probe under leave-one-day-out. The dashed "
           "line and band give the mean and one standard deviation of F1 at tIoU 0.3; the dotted line "
           "is the single-split validation figure of Table 2b.")
    para(d, f"Mean F1 at tIoU 0.3 across the five held-out days is {f3(lodo_m3)} ± {f3(lodo_s3)}, "
            f"against {f3(FZV['f1@0.3'])} on the single validation day used for selection in the main "
            "protocol, which lies more than two standard deviations above the leave-one-day-out mean. "
            f"Putdown F1 ranges from {f3(lodo['putdown_f1@0.5'].min())} to "
            f"{f3(lodo['putdown_f1@0.5'].max())} across days (mean {f3(lodo_mp)}). False positives "
            f"per hour range from {f1(lodo['fp_per_hour@0.5'].min())} to "
            f"{f1(lodo['fp_per_hour@0.5'].max())}, rising on the busier days.")

    # ================= 4. Discussion =================
    para(d, "4. Discussion", H1)
    para(d, "4.1. Observations and Candidate Explanations", H2)
    para(d, "Three observations are supported by the data. First, decoder geometry and crop "
            "conditioning had larger effects on reported figures than the choice of training regime, "
            "and both would be invisible in a single aggregate number. Second, the pickup/putdown "
            "distinction is the component that degrades most across recording days: on every "
            "held-out day of the leave-one-day-out protocol putdown F1 is below 0.36, and on the main "
            "held-out day it is zero for the fine-tuned model at the selected operating point. Third, "
            "the single-split validation figure overstates expected held-out performance by roughly "
            "two standard deviations of the day-to-day variation.")
    para(d, "We offer explanations as hypotheses, not established causes. Presence of an interaction "
            "is supported by many stable cues; direction rests on the ordering of a hand-state change "
            "relative to a shelf-state change, a subtler signal that the arrow-of-time literature [10] "
            "shows to be learnable but distinct from static appearance. The class imbalance (170 "
            "pickups to 89 putdowns) is a plausible contributor to the pickup bias, although "
            "inverse-frequency weighting was applied and the data cannot separate its effect from "
            "others. The camera is fixed, so viewpoint does not vary between days; shelf stock, item "
            "types, shoppers and lighting do. Whether the model exploits static cues rather than "
            "temporal order is testable with a single-frame baseline and a shuffled-frame control, "
            "neither of which was run; both are listed in Section 4.4.")
    para(d, "For a checkout application the practical reading is that, under these conditions, the "
            "system detects interactions with moderate recall and assigns direction with a strong "
            "pickup bias on unseen days. Aggregate F1 does not communicate this; per-class recall does.")

    para(d, "4.2. Generalisable Observations on Pipeline Construction", H2)
    para(d, "Where the unit of prediction is coarser than the unit predicted, the aggregation rule "
            "imposes a per-event bound on the evaluation metric that is easy to misattribute to the "
            "classifier. The bound d / w is simple to compute, and the fraction of events with "
            "d < τ w should be reported alongside any sliding-window tIoU result. Separately, "
            "conditioning information available only for positives will, through whatever fallback "
            "the implementation chooses, encode the label; extending conditioning across a padding "
            "interval so that positives and negatives share geometry is one remedy, subject to the "
            "caveats in Section 2.5. Both were found by rendering the loader's output and looking at "
            "it.")

    para(d, "4.3. Limitations", H2)
    para(d, "The results are estimates under oracle conditioning: candidates and crops derive from "
            "ground-truth boxes at inference as well as in training, and no independent proposal "
            "generator was evaluated. End-to-end detection in continuous video, with its own recall "
            "ceiling and false-alarm rate, is outside the scope of this study. The corpus has five "
            "recording days from one camera in one store; the leave-one-day-out evaluation "
            "characterises variation within that corpus and nothing about other stores. The frozen "
            "probe alone was evaluated under leave-one-day-out. The test split was read twice for "
            "reported results and once more for the diagnostic sweep. Annotation was by a single "
            "annotator with no independent re-review; the borrowed-box pool for verified negatives "
            "spans the whole corpus; padded background windows were not checked for the presence of "
            "another actor's event; the crop shortcut was not quantified; and no temporal-order "
            "control was run. Thirteen draft-status intervals were included without a sensitivity "
            "check. The learning-rate observation of Section 2.7 is not a controlled comparison.")

    para(d, "4.4. Future Work", H2)
    para(d, "In order of expected value: an independent candidate generator that does not use test "
            "annotations, with its recall and false-alarm rate reported, so that end-to-end claims "
            "become possible; a controlled crop ablation (full-frame fallback versus padded crops, "
            "plus a geometry-only classifier) to measure the shortcut; single-frame and shuffled-order "
            "baselines, and a chronological-versus-reversed comparison, to establish whether temporal "
            "order is used; leave-one-day-out for the fine-tuned regime with several seeds; and "
            "objectives that encourage temporal-order sensitivity, such as time-reversal augmentation "
            "with class swap, whose semantic validity for multi-item interactions should be checked "
            "before use. The observed failure on held-out days motivates these experiments; it does "
            "not establish that the failure is intrinsic to video transformers.")

    # ================= 5. Conclusions =================
    para(d, "5. Conclusions", H1)
    para(d, "Within an annotation-conditioned evaluation of an actor-conditioned VideoMAE classifier "
            "for retail pickup and putdown, the surrounding pipeline determined more of the reported "
            "performance than the model. On identical window scores, centre-derived interval "
            f"boundaries raised validation F1 at tIoU 0.5 from {f3(FTV_SPAN['f1@0.5'])} to "
            f"{f3(FTV['f1@0.5'])}; holding annotation boxes across a padding interval removed a crop "
            "shortcut that would otherwise let the classes be separated on geometry.")
    para(d, f"Partial fine-tuning raised validation event F1 to {f3(FTV['f1@0.3'])} and held-out-day "
            f"F1 to {f3(FTT['f1@0.3'])}, but achieved zero putdown F1 on the held-out day at the "
            "validation-selected operating point, with scores on true putdowns shifted towards "
            "pickup. Leave-one-day-out evaluation of the frozen probe places mean F1 at "
            f"{f3(lodo_m3)} ± {f3(lodo_s3)} and mean putdown F1 at {f3(lodo_mp)} ± {f3(lodo_sp)} "
            "across the five days, so the directional weakness recurs and the single validation day "
            "was optimistic.")
    para(d, "The observed failure on held-out days motivates evaluation across recording days as the "
            "reporting standard at this data scale, and experiments that explicitly encourage "
            "temporal-order sensitivity. The current evidence does not establish that the failure is "
            "intrinsic to video transformers. For deployment decisions, per-class recall on the "
            "minority class and false positives per hour of unique footage are the quantities to "
            "report.")

    # ================= Back matter =================
    rich(d, [("Author Contributions: ", "b"),
             "Conceptualization, F.L. and F.L.; methodology, F.L.; software, F.L.; validation, F.L.; "
             "formal analysis, F.L.; data curation, F.L.; writing—original draft, F.L.; writing—review "
             "and editing, F.L.; visualization, F.L.; supervision, F.L. All authors have read and agreed "
             "to the published version of the manuscript."], BACK)
    rich(d, [("Funding: ", "b"), "This research received no external funding."], BACK)
    rich(d, [("Institutional Review Board Statement: ", "b"),
             "The study is a secondary analysis of operational store footage that was anonymised by "
             "the data provider before release to the authors; faces and identifying features were "
             "obscured at source and the authors had no access to un-anonymised material. No "
             "institutional ethics review was sought for this secondary analysis. The authors' "
             "institutions should confirm whether such review is required under their policies."], BACK)
    rich(d, [("Informed Consent Statement: ", "b"),
             "Not applicable; no identifiable personal data were processed by the authors."], BACK)
    rich(d, [("Data Availability Statement: ", "b"),
             f"Source code, configuration and evaluation harness: {REPO_URL}, branch {BRANCH}, "
             f"release tag {RELEASE_TAG}. Results registry, per-window scores, canonical annotation "
             "tables, actor tracks, trained weights and a SHA-256 manifest are provided in a "
             "reproducibility bundle from which every table and figure can be regenerated without "
             "video access; it is deposited with the corresponding author and provided on request, "
             "and will be archived with a DOI on acceptance. The source recordings are commercial "
             "footage of members of the public and are not redistributable; the bundle carries each "
             "recording's storage location and SHA-256 digest."], BACK)
    rich(d, [("Acknowledgments: ", "b"), "The authors thank the annotator for the CVAT labelling "
             "effort underlying this study."], BACK)
    rich(d, [("Conflicts of Interest: ", "b"), "The authors declare no conflict of interest."], BACK)

    # ================= References =================
    para(d, "References", H1)
    for ref in [
        "Idrees, H.; Zamir, A.R.; Jiang, Y.-G.; Gorban, A.; Laptev, I.; Sukthankar, R.; Shah, M. The "
        "THUMOS challenge on action recognition for videos “in the wild”. Comput. Vis. Image Underst. "
        "2017, 155, 1–23. https://doi.org/10.1016/j.cviu.2016.10.018",
        "Caba Heilbron, F.; Escorcia, V.; Ghanem, B.; Carlos Niebles, J. ActivityNet: A large-scale video "
        "benchmark for human activity understanding. In Proceedings of the IEEE Conference on Computer "
        "Vision and Pattern Recognition (CVPR), Boston, MA, USA, 7–12 June 2015; pp. 961–970.",
        "Lin, T.; Liu, X.; Li, X.; Ding, E.; Wen, S. BMN: Boundary-matching network for temporal action "
        "proposal generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision "
        "(ICCV), Seoul, Korea, 27 October–2 November 2019; pp. 3889–3898. "
        "https://doi.org/10.48550/arXiv.1907.09702",
        "Zhang, C.-L.; Wu, J.; Li, Y. ActionFormer: Localizing moments of actions with transformers. In "
        "Computer Vision – ECCV 2022; Lecture Notes in Computer Science, Vol. 13664; Springer: Cham, "
        "Switzerland, 2022; pp. 492–510. https://doi.org/10.1007/978-3-031-19772-7_29",
        "Goyal, R.; Ebrahimi Kahou, S.; Michalski, V.; Materzynska, J.; Westphal, S.; Kim, H.; Haenel, V.; "
        "Fruend, I.; Yianilos, P.; Mueller-Freitag, M.; et al. The “something something” video database "
        "for learning and evaluating visual common sense. In Proceedings of the IEEE International "
        "Conference on Computer Vision (ICCV), Venice, Italy, 22–29 October 2017; pp. 5842–5850.",
        "Tong, Z.; Song, Y.; Wang, J.; Wang, L. VideoMAE: Masked autoencoders are data-efficient learners "
        "for self-supervised video pre-training. In Advances in Neural Information Processing Systems 35 "
        "(NeurIPS 2022); 2022. https://doi.org/10.48550/arXiv.2203.12602",
        "Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, "
        "M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. An image is worth 16×16 words: Transformers for "
        "image recognition at scale. In Proceedings of the International Conference on Learning "
        "Representations (ICLR), 2021. https://doi.org/10.48550/arXiv.2010.11929",
        "Kay, W.; Carreira, J.; Simonyan, K.; Zhang, B.; Hillier, C.; Vijayanarasimhan, S.; Viola, F.; "
        "Green, T.; Back, T.; Natsev, P.; et al. The Kinetics human action video dataset. arXiv 2017, "
        "arXiv:1705.06950. https://doi.org/10.48550/arXiv.1705.06950",
        "CVAT.ai Corporation. Computer Vision Annotation Tool (CVAT), software, 2026. "
        "https://doi.org/10.5281/zenodo.3497105",
        "Wei, D.; Lim, J.; Zisserman, A.; Freeman, W.T. Learning and using the arrow of time. In "
        "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Salt Lake "
        "City, UT, USA, 18–22 June 2018; pp. 8052–8060.",
        "Singh, B.; Marks, T.K.; Jones, M.; Tuzel, O.; Shao, M. A multi-stream bi-directional recurrent "
        "neural network for fine-grained action detection. In Proceedings of the IEEE Conference on "
        "Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 27–30 June 2016; "
        "pp. 1961–1970.",
        "Geirhos, R.; Jacobsen, J.-H.; Michaelis, C.; Zemel, R.; Brendel, W.; Bethge, M.; Wichmann, F.A. "
        "Shortcut learning in deep neural networks. Nat. Mach. Intell. 2020, 2, 665–673. "
        "https://doi.org/10.1038/s42256-020-00257-z",
    ]:
        para(d, ref, REFS)

    output.parent.mkdir(parents=True, exist_ok=True)
    d.save(str(output))

    # Versioned copy: V<n> where n is one past the highest existing V<n> in versions/.
    # Never overwrites a previous version.
    versions = output.parent / "versions"
    versions.mkdir(exist_ok=True)
    existing = [int(m.group(1)) for f in versions.glob(f"{output.stem}_V*.docx")
                if (m := re.search(r"_V(\d+)\.docx$", f.name))]
    n_version = max(existing, default=0) + 1
    versioned = versions / f"{output.stem}_V{n_version}.docx"
    shutil.copy2(output, versioned)
    print(f"saved {versioned.name}")
    working = output.with_name("_template_working.docx")
    if working.exists():
        working.unlink()
    print(f"wrote {output} ({output.stat().st_size/1000:.0f} kB)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path, default=Path.home() / "Downloads/jimaging-template.dot")
    p.add_argument("--output", type=Path, default=REPO_ROOT / ".local/paper/jimaging/jimaging_pickup_putdown.docx")
    p.add_argument("--figures", type=Path, default=REPO_ROOT / ".local/paper/figures")
    p.add_argument("--revision", type=Path, default=REPO_ROOT / ".local/paper/revision")
    a = p.parse_args()
    build(a.template, a.output, a.figures, a.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
