#!/usr/bin/env python3
"""Build the Journal of Imaging manuscript from MDPI's official template.

    python scripts/make_jimaging_paper.py

Fills the template in place: placeholder body content is removed, ours is written
back using the template's own MDPI_* named styles, so page setup, headers, footers,
fonts and spacing are whatever MDPI shipped.

The template is distributed as .dot with a template content type that python-docx
refuses; it is rewritten to a document content type on load. Nothing else is altered.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/lubomir-angelov/ai_workshop_cv_case"
BRANCH = "feature/cvat_annotation"

TEXT = "MDPI_3.1_text"
TEXT_NI = "MDPI_3.2_text_no_indent"
H1, H2, H3 = "MDPI_2.1_heading1", "MDPI_2.2_heading2", "MDPI_2.3_heading3"
BACK = "MDPI_6.2_back_matter"
REFS = "MDPI_8.1_references"
FIGCAP = "MDPI_5.1_figure_caption"
TABCAP = "MDPI_4.1_table_caption"


def open_template(source: Path, working: Path) -> Document:
    """Copy the .dot to .docx and relax its content type so python-docx will open it."""
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
    """Drop template placeholder content, keeping section properties (margins, headers)."""
    body = document.element.body
    for child in list(body):
        if not child.tag.endswith("}sectPr"):
            body.remove(child)


def para(document: Document, text: str, style: str = TEXT):
    p = document.add_paragraph(style=style)
    p.add_run(text)
    return p


def rich(document: Document, parts, style: str = TEXT):
    """Paragraph with mixed formatting: str, or (text, 'b'|'i') pairs."""
    p = document.add_paragraph(style=style)
    for part in parts:
        text, mark = part if isinstance(part, tuple) else (part, "")
        run = p.add_run(text)
        run.bold = "b" in mark
        run.italic = "i" in mark
    return p


def figure(document: Document, image: Path, caption: str, width_in: float = 6.1) -> None:
    if not image.exists():
        return
    p = document.add_paragraph(style="MDPI_5.2_figure")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(image), width=Inches(width_in))
    cap = document.add_paragraph(style=FIGCAP)
    head, _, rest = caption.partition(". ")
    cap.add_run(head + ". ").bold = True
    cap.add_run(rest)


def table(document: Document, rows: list[list[str]], caption: str) -> None:
    cap = document.add_paragraph(style=TABCAP)
    head, _, rest = caption.partition(". ")
    cap.add_run(head + ". ").bold = True
    cap.add_run(rest)

    t = document.add_table(rows=len(rows), cols=len(rows[0]))
    for style_name in ("MDPI_4.1_three_line_table", "MDPI_table", "Table Grid"):
        try:
            t.style = style_name
            break
        except KeyError:
            continue
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = t.cell(r, c)
            cell.text = ""
            p = cell.paragraphs[0]
            p.style = document.styles["MDPI_4.2_table_body"]
            if c:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(value))
            run.bold = r == 0
            run.font.size = Pt(8)


def metrics() -> dict:
    def read(path):
        p = REPO_ROOT / path
        return json.loads(p.read_text()) if p.exists() else {}
    return {
        "ft_val": read(".local/track_b1_finetune/predictions/metrics_val.json"),
        "ft_test": read(".local/track_b1_finetune/predictions/metrics_test.json"),
        "fz_val": read(".local/track_b1_run_w15/predictions/metrics_val.json"),
        "fz_test": read(".local/track_b1_run_w15/predictions/metrics_test.json"),
    }


def num(node: dict, *path, fmt="{:.3f}", default="—"):
    for step in path:
        node = node.get(step) if isinstance(node, dict) else None
    return fmt.format(node) if isinstance(node, (int, float)) else default


def build(template: Path, output: Path, figures: Path) -> None:
    m = metrics()
    d = open_template(template, output.with_name("_template_working.docx"))
    clear_body(d)

    # ---------------- Front matter ----------------
    para(d, "Article", "MDPI_1.1_article_type")
    para(d, "Actor-Conditioned Video Transformers for Pickup and Putdown Detection in "
            "Retail Video: Decoder Geometry, Crop Conditioning and Cross-Day "
            "Generalisation", "MDPI_1.2_title")
    rich(d, ["Firstname Lastname ", ("1", "i"), ", Firstname Lastname ", ("2", "i"),
             " and Firstname Lastname ", ("1,*", "i")], "MDPI_1.3_authornames")
    para(d, "1\tAffiliation 1; e-mail@e-mail.com", "MDPI_1.6_affiliation")
    para(d, "2\tAffiliation 2; e-mail@e-mail.com", "MDPI_1.6_affiliation")
    para(d, "*\tCorrespondence: e-mail@e-mail.com", "MDPI_1.6_affiliation")

    rich(d, [("Abstract: ", "b"),
             "Distinguishing an item being removed from a shelf from one being returned to it is a "
             "temporal action detection problem in which the two classes are near time-reverses of "
             "one another, so the discriminating evidence is the direction of transfer rather than "
             "its presence. We train an actor-conditioned VideoMAE window classifier on 42 "
             "human-annotated 4K store recordings comprising 259 events across five recording days, "
             "and evaluate it with a class-aware temporal intersection-over-union protocol. Two "
             "results concern the pipeline rather than the model. Constructing a predicted "
             "interval as the span of contributing sliding windows imposes a ceiling on "
             "achievable temporal IoU equal to the ratio of event to window duration; a "
             "centre-derived interval raised validation F1 at tIoU 0.3 from 0.350 to 0.647 "
             "without retraining. Where spatial conditioning exists only for positives, "
             "background windows fall back to a full-frame crop and the classes become separable "
             "on crop geometry alone. The substantive result is negative: fine-tuning the last two "
             "transformer blocks raised validation event F1 to 0.783, but on a held-out recording "
             "day mean predicted putdown probability on true putdown windows fell to 0.062 while "
             "pickup rose to 0.588, and no threshold recovers the class. Detection and localisation "
             "transfer across days; direction does not."],
         "MDPI_1.7_abstract")
    rich(d, [("Keywords: ", "b"),
              "temporal action detection; video transformers; VideoMAE; retail computer vision; "
              "domain shift; annotation methodology; evaluation protocol; self-checkout"],
         "MDPI_1.8_keywords")

    # ---------------- 1. Introduction ----------------
    para(d, "1. Introduction", H1)
    para(d, "Automated checkout and shrinkage analytics both depend on a single perceptual "
            "distinction: whether an item moved from the shelf into a shopper's possession, or in "
            "the opposite direction. A system that reports only that an interaction occurred is not "
            "useful for either application. Charging a customer for an item they inspected and "
            "returned is a worse outcome than failing to detect the event, because the error is "
            "visible to the customer, adversarial in character, and costly to dispute.")
    para(d, "This asymmetry makes the task an awkward fit for the standard temporal action "
            "detection formulation. Established benchmarks such as THUMOS14 [1] and ActivityNet [2] "
            "contain action classes that differ in object, scene and appearance, so a model can "
            "succeed on largely static cues. The methods that lead those benchmarks reflect that "
            "assumption: proposal-based detectors such as BMN [3] and single-stage transformer "
            "detectors such as ActionFormer [4] localise segments whose class identity is settled "
            "principally by segment content. Pickup and putdown share every such cue. They occur at "
            "the same shelf, involve the same hands and merchandise, and produce near-identical "
            "spatial evidence. What separates them is the temporal ordering of a hand-state change "
            "relative to a shelf-state change. The Something-Something dataset [5] was constructed "
            "specifically to expose this weakness in appearance-driven models, and retail "
            "pickup/putdown is a real-world instance of the same difficulty.")
    para(d, "Video transformers pre-trained by masked autoencoding [6], building on the Vision "
            "Transformer architecture [7] and large-scale action corpora [8], are now the default "
            "backbone for such tasks. Data efficiency on small downstream datasets is an explicit "
            "claim of VideoMAE [6], which reports competitive results on corpora of three to four "
            "thousand clips. Whether that efficiency extends to a distinction defined by temporal "
            "direction, under the domain shift induced by a different recording day in the same "
            "store, is not something existing benchmarks measure.")
    para(d, "We report a case study addressing that question. The contribution is deliberately "
            "narrow and empirical, and consists of three parts. First, a data path from "
            "source-video box annotation to actor-conditioned window training that requires no "
            "pose-estimation stage: annotators draw a tracked box on each interaction in CVAT [9], "
            "and that single artefact supplies both the temporal extent of the event and the "
            "spatial region on which the classifier is conditioned. Second, two failure modes in "
            "the machinery surrounding the model which are straightforward to introduce, invisible "
            "in aggregate accuracy, and large enough to dominate reported performance; one concerns "
            "the construction of predicted intervals from window scores, the other the behaviour of "
            "the crop when no annotation box is available. Third, a quantified negative result on "
            "the cross-day generalisation of the direction distinction, accompanied by the "
            "diagnostic evidence required to separate it from a threshold-calibration failure.")
    para(d, "The second of these is the contribution we expect to transfer beyond this dataset. "
            "Both failure modes are properties of pipeline construction rather than of the "
            "architecture, both proved more consequential than any modelling change we made, and "
            "neither is visible in a single headline metric.")

    # ---------------- 2. Materials and Methods ----------------
    para(d, "2. Materials and Methods", H1)
    para(d, "This section describes each stage of the pipeline in the order it is executed. The "
            "complete implementation, configuration files and evaluation code are openly available "
            f"(Section 2.9). Figure 1 of the software documentation and the command "
            f"make track-b1-all reproduce the entire sequence from an existing annotation export.")

    para(d, "2.1. Video Corpus and Acquisition", H2)
    para(d, "The corpus comprises 42 recordings from a single fixed overhead camera in one retail "
            "store. Each recording is between two and five minutes long, at a spatial resolution of "
            "3840 × 2160 pixels and a frame rate of exactly 20.0 frames per second, and the set "
            "spans five distinct recording days. Faces and other identifying features were "
            "anonymised at source before the footage was made available to the annotation team, and "
            "no personally identifying information was accessible at any subsequent stage.")
    para(d, "Table 1 summarises the corpus and the split assignment described in Section 2.8.")
    table(d, [
        ["Split", "Recording days", "Clips", "Pickup", "Putdown", "Windows"],
        ["Train", "3 (20–22 May)", "23", "127", "65", "5192"],
        ["Validation", "1 (23 May)", "7", "20", "12", "973"],
        ["Test", "1 (26 May)", "12", "23", "12", "1551"],
        ["Total", "5", "42", "170", "89", "7716"],
    ], "Table 1. Composition of the annotated corpus and assignment of recordings to splits. "
       "Splits are assigned at the level of recording day, never at the level of individual clip.")

    para(d, "2.2. Annotation Protocol", H2)
    para(d, "Annotation was performed in CVAT [9] directly on the source video rather than on "
            "pre-extracted candidate windows. Each annotated interval is a CVAT track carrying an "
            "axis-aligned bounding box on every frame it spans, labelled pickup, putdown or ignore. "
            "Per-label attributes record annotator confidence (high, medium or low), a hard-case "
            "flag, an item count, and a review status. A track interrupted by an outside marker and "
            "subsequently resumed is treated as two distinct events rather than a single interval "
            "spanning the intervening gap.")
    para(d, "The resulting ground truth contains 259 events, of which 170 are pickup and 89 are "
            "putdown, together with three ignore intervals marking spans in which transfer evidence "
            "was unavailable. The median event duration is 0.90 s and the median annotation box "
            "measures 214 × 249 pixels within the 3840 × 2160 frame. Seven of the 42 recordings "
            "were completed by an annotator with zero events; these are verified negatives and are "
            "retained, since a confirmed absence of interaction is as informative for training as a "
            "confirmed presence.")
    para(d, "The design choice to annotate on source video rather than on candidate windows has two "
            "consequences that shape the remainder of the method. It removes any dependence on a "
            "pose-estimation stage, because the annotation box itself identifies the region of "
            "interest; and it makes the annotation the sole source of spatial conditioning, which "
            "introduces the limitation discussed in Section 4.3.")

    para(d, "2.3. Step 1: Recovery of a Canonical Representation", H2)
    para(d, "CVAT stores annotations as frame indices, so every derived timestamp depends on the "
            "assumed frame rate. Estimating that rate from the recording window encoded in each "
            "filename yields approximately 20.04 fps, which differs from the true rate by "
            "approximately 0.2%. Accumulated over a three-minute recording this produces a drift of "
            "roughly 0.35 s, which exceeds the median event duration. We therefore probe the exact "
            "average frame rate directly from each video file and record the provenance of the "
            "value, so that an estimated rate is never mistaken for a measured one. All 42 "
            "recordings are exactly 20.0 fps.")
    para(d, "Each archive is parsed into four canonical tables: events, with type, start and end "
            "times, confidence and hard-case flag; ignore intervals; clip metadata including frame "
            "rate, duration and split assignment; and per-frame actor tracks giving the annotation "
            "box at every frame. The actor-track table adopts the column layout of a pose-track "
            "table, which allows the downstream dataset to consume it without modification.")

    para(d, "2.4. Step 2: Actor-Conditioned Candidates and Windows", H2)
    para(d, "Each CVAT track is treated as one actor stream. Because tracks are drawn per "
            "interaction rather than per person, two temporally overlapping tracks within a "
            "recording yield two independent actor streams; the corpus contains 273 such "
            "overlapping pairs. Window labels are assigned per actor, using only those events "
            "belonging to that actor's stream. Assigning labels at the level of the recording, "
            "which is the more obvious implementation, associates one actor's pickup with a window "
            "cropped around a different actor and introduces label noise directly into training.")
    para(d, "A candidate is the padded temporal extent of one actor track. Sliding windows of fixed "
            "duration are generated across each candidate at fixed stride, and each window is "
            "labelled according to what occupies its centre: the event class if the centre falls "
            "within an event interval belonging to that actor, and background otherwise. Windows "
            "whose centre falls within an ignore interval are discarded. Sample weights follow "
            "annotator confidence, with low-confidence events down-weighted by a factor of two.")

    para(d, "2.5. Step 3: Crop Conditioning and the Padding Requirement", H2)
    para(d, "The annotation box defines the crop. For a given candidate, the crop region is the "
            "union of that actor's boxes across the candidate, expanded by a margin of 15% and "
            "clamped to the frame boundary, then resized to 224 × 224 pixels.")
    para(d, "Annotation boxes exist only while an event is in progress. A background window drawn "
            "from outside any event therefore finds no box. The natural implementation fallback, "
            "cropping to the full frame, is damaging in a manner that does not appear in the "
            "training loss: every positive example becomes a tight crop around a pair of hands and "
            "every negative a wide-angle 4K view, so the classes become separable on crop geometry "
            "alone and a model may attain high accuracy without attending to the action at all.")
    para(d, "We remove this shortcut by extending each actor track by ±3 s before and after the "
            "annotated interval, repeating the boundary box across the extension. Background "
            "windows drawn from this padding are cropped identically to the positive windows "
            "originating from the same candidate, so that only the motion contained within the "
            "region distinguishes them. The padding interval additionally supplies the hardest "
            "available negatives, since it contains the approach and withdrawal that immediately "
            "surround a genuine interaction.")
    para(d, "For the seven verified-negative recordings no annotation box exists anywhere. Twelve "
            "candidates per recording are synthesised at randomly sampled positions, each "
            "conditioned on a box drawn from the pool of real annotation boxes, so that the "
            "crop-geometry distribution of these negatives matches that of the positive class. The "
            "borrowed box does not correspond to any person's actual location in that recording; "
            "these windows are consequently valid negatives for the question the classifier is "
            "asked, namely whether a transfer is occurring within the crop, but not for the "
            "question of whether a person is present.")
    para(d, "The crop region is held constant across all windows within a candidate. Besides "
            "enabling the frame cache described in Section 2.6, this is the preferable conditioning "
            "choice: a crop that varied per window would expand and contract with the actor's "
            "motion, and that variation is correlated with event timing.")

    para(d, "2.6. Step 4: Frame Caching", H2)
    para(d, "The source recordings are 4K H.264. Decoding a training window by seeking to each of "
            "its 16 constituent frames is prohibitively slow, because every seek decodes forward "
            "from the preceding keyframe at full resolution. Measured on one recording, 16 seeking "
            "reads required 20.6 s whereas 48 sequential reads required 0.4 s, a per-frame "
            "difference of approximately two orders of magnitude. At roughly 27 s per window, a "
            "single training epoch over 5192 windows would require approximately seventeen hours.")
    para(d, "Each candidate is therefore decoded exactly once, sequentially, cropped to its "
            "actor-conditioned region, resized and stored as an unsigned 8-bit array; windows are "
            "subsequently served by indexing into memory-mapped arrays. The cache occupies 7.2 GB "
            "and is constructed in approximately twenty minutes.")
    para(d, "A second optimisation applies while the encoder is frozen. In that regime the backbone "
            "contributes 86.2 M parameters that are recomputed every epoch in order to train 3843, "
            "and its output is invariant across epochs. Computing the 768-dimensional embeddings "
            "once for the full manifest reduces an epoch from approximately nine minutes to under "
            "one second, which is what makes an exhaustive threshold search and genuine early "
            "stopping affordable. This optimisation is valid only while the backbone is frozen.")

    para(d, "2.7. Step 5: Model and Training Regimes", H2)
    para(d, "We use VideoMAE-Base [6] pre-trained on Kinetics-400 [8], with a classification head "
            "consisting of layer normalisation, dropout and a linear projection to three classes "
            "(background, pickup, putdown). Each window is represented by 16 frames sampled "
            "uniformly in chronological order and normalised with ImageNet statistics; encoder "
            "outputs are mean-pooled over the sequence dimension before the head.")
    para(d, "Two regimes are compared. In the frozen-probe regime the backbone is entirely frozen "
            "and only the head is trained. In the partial fine-tuning regime the final two "
            "transformer blocks are unfrozen alongside the head, which requires the full pixel "
            "path. Two configuration details in the second regime proved necessary rather than "
            "optional. Discriminative learning rates are required: the head trains at 1 × 10⁻³ and "
            "the unfrozen backbone at 5 × 10⁻⁵, and a single rate applied to both fails in either "
            "direction. In an early run at a uniform 5 × 10⁻⁵ the randomly initialised head was "
            "still predicting background exclusively after two epochs, an outcome we initially "
            "misread as evidence that unfreezing does not help. The head is additionally "
            "warm-started from the trained frozen probe, so that the comparison measures the "
            "contribution of unfreezing rather than the rate at which a head can be trained from "
            "random initialisation.")
    para(d, "Both regimes use inverse-frequency class weights, cosine learning-rate decay with "
            "warmup, and checkpoint selection on validation macro F1.")
    para(d, "Before any full training run we require two verification gates to pass. The first "
            "renders the frames produced by the data loader as index-stamped grids, so that "
            "temporal ordering and crop correctness are verified visually. The second is a "
            "tiny-overfit test on a class-balanced subset; class balance is material, because "
            "drawing the leading rows of the manifest yields samples from a single candidate and "
            "usually a single class, which a model can memorise by collapsing to a constant "
            "prediction, allowing the gate to pass without evidence of a functioning pipeline.")

    para(d, "2.8. Step 6: Inference, Decoding and Evaluation", H2)
    para(d, "At inference, windows slide across each candidate, class probabilities are smoothed "
            "over adjacent windows, contiguous runs exceeding a per-class threshold become score "
            "regions, and only regions of the same type are merged. A pickup adjacent to a putdown "
            "is never merged into a single event, and one candidate may emit zero, one or several "
            "ordered events.")
    para(d, "Converting a run of above-threshold windows into an interval admits two natural "
            "definitions, and the choice is consequential. Under the window-span definition the "
            "interval runs from the first contributing window's start to the last one's end, so a "
            "single above-threshold window produces an interval exactly one window in length. Under "
            "the window-centre definition the interval runs from the first to the last contributing "
            "window's centre, widened by half a stride on each side and subject to a minimum "
            "duration.")
    para(d, "The window-span definition carries a structural ceiling. If a single window fires on an "
            "event of duration d with window length w, the predicted interval has length w and the "
            "temporal IoU against the ground truth cannot exceed d / w. With d = 0.90 s and "
            "w = 2.5 s that ceiling is 0.36, which lies below the conventional 0.5 operating point "
            "irrespective of classifier quality. The window-centre definition carries no such "
            "ceiling and is better motivated on its own terms, since a window is labelled by what "
            "occupies its centre and the centre is therefore where its temporal evidence resides.")
    para(d, "Predictions are scored with a class-aware evaluator that applies ignore intervals "
            "consistently to both ground truth and predictions. We report precision, recall and F1 "
            "at tIoU thresholds of 0.3 and 0.5, per-class F1, mean average precision, boundary mean "
            "absolute error, and false positives per hour. Matching is one-to-one by Hungarian "
            "assignment, and a prediction matches a ground-truth event only if the classes agree.")
    para(d, "Splits are assigned at the level of recording day rather than clip. Recordings made "
            "minutes apart share shoppers, lighting, shelf stock and merchandise placement, so a "
            "clip-level split would place near-duplicates on both sides of the boundary and inflate "
            "validation performance. Days are ranked by event count and the sparsest held out, "
            "retaining the bulk of supervision in training; the dataset builder fails if any "
            "recording appears under two splits. Decision thresholds and the smoothing width are "
            "selected on validation alone, by the mean of F1 at tIoU 0.3 and 0.5 over a grid of 200 "
            "combinations. The test split was not read during any selection step.")

    para(d, "2.9. Code, Data and Reproducibility", H2)
    para(d, "The complete implementation is openly available in the project repository at "
            f"{REPO_URL} (branch {BRANCH}). The repository contains the CVAT import and canonical "
            "table construction, the actor-conditioned window builder, the frame and embedding "
            "caches, both training entry points, the sliding-window inference and decoding stage, "
            "the threshold search, and the evaluation harness, together with 47 unit tests covering "
            "the annotation import, window labelling, caching and decoding logic. Method "
            "documentation is provided in docs/TRACK_B1_CVAT.md and the configuration used for the "
            "reported acceptance run in configs/track_b1.yaml. The full pipeline is reproduced from "
            "an existing annotation export by the single command make track-b1-all.")
    para(d, "A reproducibility bundle accompanies the work and contains the 42 CVAT annotation "
            "archives exactly as exported, the canonical event, clip, candidate and window-manifest "
            "tables in both Parquet and CSV form, the per-clip actor tracks, the trained model "
            "weights for both regimes, the complete evaluation artefacts for every reported "
            "configuration, and a SHA-256 manifest covering every file. The source recordings "
            "themselves are commercial store footage of members of the public and are not "
            "redistributed; the bundle instead carries each recording's object-store location, size "
            "and SHA-256 digest, so that a holder of the appropriate credentials can reconstruct the "
            "exact input set and verify it byte-for-byte. Every figure and table in this manuscript "
            "is generated directly from the stored evaluation artefacts rather than transcribed, "
            "and the generating scripts are included in the repository.")

    # ---------------- 3. Results ----------------
    para(d, "3. Results", H1)

    para(d, "3.1. Decoder Geometry Dominates the Ablation", H2)
    para(d, "Figure 1 reports validation F1 across four decode configurations with the frozen-probe "
            "model held fixed. Tuning decision thresholds under window-span boundaries raises F1 at "
            "tIoU 0.3 from 0.264 to 0.350. Substituting centre-derived boundaries raises it to "
            "0.647, an improvement approximately four times larger than that obtained from "
            "threshold tuning and achieved without retraining any component.")
    figure(d, figures / "fig1_decode_ablation.png",
           "Figure 1. Validation event-level F1 across four decode configurations for the "
           "frozen-probe model. The model is identical in all four; only the conversion of window "
           "scores into intervals differs. The tIoU 0.5 bar is omitted for the second configuration, "
           "which was not evaluated at that threshold.")
    para(d, "The effect is most pronounced at the stricter operating point. Under window-span "
            "boundaries, F1 at tIoU 0.5 was 0.075, consistent with the predicted ceiling of 0.36 "
            "for 2.5 s windows applied to 0.9 s events. Under window-centre boundaries with a 1.5 s "
            "window it reaches 0.523, and boundary mean absolute error falls from 1.35 s to 0.21 s. "
            "Shortening the window from 2.5 s to 1.5 s and the stride from 0.5 s to 0.25 s "
            "contributes a further gain concentrated at tIoU 0.5, which is the expected pattern if "
            "the residual error after the boundary correction is temporal resolution rather than "
            "classification.")

    para(d, "3.2. Partial Fine-Tuning Improves Validation Performance", H2)
    para(d, "Table 2 compares the two training regimes on validation. Unfreezing the final two "
            "transformer blocks improves every reported metric, with the largest gains on the event "
            "classes rather than on background.")
    table(d, [
        ["Metric", "Frozen probe", "Fine-tuned (2 blocks)"],
        ["Window-level macro F1", "0.615", "0.743"],
        ["Event F1 @ tIoU 0.3", num(m["fz_val"], "tiou@0.3", "f1"), num(m["ft_val"], "tiou@0.3", "f1")],
        ["Event F1 @ tIoU 0.5", num(m["fz_val"], "tiou@0.5", "f1"), num(m["ft_val"], "tiou@0.5", "f1")],
        ["Pickup F1", num(m["fz_val"], "per_type", "pickup", "f1"), num(m["ft_val"], "per_type", "pickup", "f1")],
        ["Putdown F1", num(m["fz_val"], "per_type", "putdown", "f1"), num(m["ft_val"], "per_type", "putdown", "f1")],
        ["mAP (average)", num(m["fz_val"], "mAP", "mAP_avg"), num(m["ft_val"], "mAP", "mAP_avg")],
        ["Boundary MAE, start (s)", num(m["fz_val"], "start_mae_s", fmt="{:.2f}"),
         num(m["ft_val"], "start_mae_s", fmt="{:.2f}")],
        ["False positives per hour", num(m["fz_val"], "fp_per_hour", fmt="{:.1f}"),
         num(m["ft_val"], "fp_per_hour", fmt="{:.1f}")],
    ], "Table 2. Validation performance of the two training regimes. Both configurations use 1.5 s "
       "windows, centre-derived interval boundaries, and decision thresholds selected on validation.")
    para(d, "The fine-tuning trajectory is not monotone. After the first epoch the model reached a "
            "pickup F1 of 0.635 while putdown collapsed to 0.057; the second epoch degraded both; "
            "convergence to the reported optimum occurred at the fourth epoch, after which the "
            "gap between training and validation performance began to widen. A schedule terminated "
            "early on a short patience would have supported the opposite conclusion.")

    para(d, "3.3. Performance on the Held-Out Recording Day", H2)
    para(d, "Table 3 reports both models on the held-out day, each evaluated with its own "
            "validation-selected configuration.")
    table(d, [
        ["Metric", "Frozen probe", "Fine-tuned (2 blocks)"],
        ["Event F1 @ tIoU 0.3", num(m["fz_test"], "tiou@0.3", "f1"), num(m["ft_test"], "tiou@0.3", "f1")],
        ["Event F1 @ tIoU 0.5", num(m["fz_test"], "tiou@0.5", "f1"), num(m["ft_test"], "tiou@0.5", "f1")],
        ["Pickup F1", num(m["fz_test"], "per_type", "pickup", "f1"), num(m["ft_test"], "per_type", "pickup", "f1")],
        ["Putdown F1", num(m["fz_test"], "per_type", "putdown", "f1"), num(m["ft_test"], "per_type", "putdown", "f1")],
        ["mAP (average)", num(m["fz_test"], "mAP", "mAP_avg"), num(m["ft_test"], "mAP", "mAP_avg")],
        ["False positives per hour", num(m["fz_test"], "fp_per_hour", fmt="{:.1f}"),
         num(m["ft_test"], "fp_per_hour", fmt="{:.1f}")],
    ], "Table 3. Performance on the held-out recording day. The configuration of each model was "
       "frozen on validation before the test split was read.")
    para(d, "Fine-tuning improves the aggregate figures and the direction of improvement agrees "
            "with validation. The magnitude does not: validation F1 at tIoU 0.3 is "
            f"{num(m['ft_val'], 'tiou@0.3', 'f1')} whereas the held-out day yields "
            f"{num(m['ft_test'], 'tiou@0.3', 'f1')}. Figure 2 shows the gap for both models and "
            "both classes.")
    figure(d, figures / "fig2_val_test_gap.png",
           "Figure 2. Validation and held-out-day event-level F1 for both training regimes. The "
           "aggregate gap is substantial for both models and is carried almost entirely by the "
           "putdown class.")

    para(d, "3.4. The Gap Is a Directional Failure, Not a Calibration Failure", H2)
    para(d, "The putdown class fails completely on the held-out day. The fine-tuned model predicted "
            "33 pickups and 2 putdowns against a ground truth of 23 and 12 respectively.")
    para(d, "The immediate hypothesis is threshold miscalibration, the putdown threshold of 0.45 "
            "selected on the validation day being too strict for the held-out day. The window-score "
            "distributions exclude this explanation. Table 4 reports mean predicted probabilities "
            "on windows whose ground-truth label is putdown.")
    table(d, [
        ["Split", "Mean p(putdown)", "Mean p(pickup)", "Fraction above threshold"],
        ["Validation day", "0.445", "0.296", "0.49"],
        ["Held-out test day", "0.062", "0.588", "0.02"],
    ], "Table 4. Mean predicted class probability on windows whose ground-truth label is putdown, "
       "fine-tuned model.")
    para(d, "On the held-out day the model is not uncertain about putdowns. It assigns them a mean "
            "pickup probability of 0.588, which is to say it confidently classifies them as the "
            "opposite class. Figure 3 shows the full distributions: on validation the scores are "
            "bimodal with substantial mass above the decision threshold, whereas on the held-out "
            "day the entire distribution lies below it. The maximum p(putdown) across all 57 "
            "true-putdown windows of the held-out day is 0.467, so no threshold choice recovers the "
            "class.")
    figure(d, figures / "fig3_putdown_probability_shift.png",
           "Figure 3. Distribution of predicted putdown probability on windows whose ground truth "
           "is putdown, fine-tuned model. On the validation day the distribution is bimodal with "
           "substantial mass above the decision threshold. On the held-out day it lies almost "
           "entirely near zero, with no window exceeding 0.467.", width_in=4.7)
    para(d, "The class-confusion structure in Figure 4 confirms this interpretation. On validation, "
            "matched putdowns are predominantly recovered as putdowns. On the held-out day every "
            "matched putdown is assigned to pickup and none to putdown.")
    figure(d, figures / "fig4_confusion.png",
           "Figure 4. Class confusion at tIoU 0.5 for the fine-tuned model. Rows denote ground "
           "truth and columns prediction. On the held-out day all six matched putdown events are "
           "classified as pickups.", width_in=4.7)
    para(d, "Localisation, by contrast, transfers acceptably. Boundary mean absolute error on the "
            f"held-out day is {num(m['ft_test'], 'start_mae_s', fmt='{:.2f}')} s at the start and "
            f"{num(m['ft_test'], 'end_mae_s', fmt='{:.2f}')} s at the end, against "
            f"{num(m['ft_val'], 'start_mae_s', fmt='{:.2f}')} s and "
            f"{num(m['ft_val'], 'end_mae_s', fmt='{:.2f}')} s on validation. The model continues to "
            "detect interactions and to place them accurately in time. What it loses across "
            "recording days is the direction.")

    # ---------------- 4. Discussion ----------------
    para(d, "4. Discussion", H1)

    para(d, "4.1. What Transfers and What Does Not", H2)
    para(d, "The results separate into two components. Detecting that a shelf interaction has "
            "occurred, and localising it in time, transfer across recording days with moderate "
            "degradation. Determining the direction in which the item moved does not transfer.")
    para(d, "This is consistent with the structure of the task. The presence of an interaction is "
            "supported by cues that are abundant and stable: a hand enters the shelf region, "
            "occlusion patterns change, motion energy rises. Direction is supported by a single and "
            "subtler cue, the temporal ordering of the hand-state change relative to the "
            "shelf-state change, and that ordering is expressed in appearance in ways that depend "
            "on camera angle, shelf geometry, item type and individual reaching behaviour. A model "
            "of sufficient capacity will fit the majority class's expression of that cue on the "
            "training days without acquiring anything that survives a change of day.")
    para(d, "The class imbalance reinforces the effect. Pickups outnumber putdowns by 170 to 89 "
            "across the corpus and by 23 to 12 on the held-out day. A model uncertain about "
            "direction minimises expected loss by defaulting to the more frequent class, which is "
            "precisely the behaviour the confusion matrix exhibits.")
    para(d, "For a self-checkout application this failure mode is more damaging than the aggregate "
            "F1 conveys. A system that detects interactions reliably but assigns direction "
            f"according to the class prior will systematically charge customers for items they "
            f"returned to the shelf. An aggregate F1 of {num(m['ft_test'], 'tiou@0.3', 'f1')} does "
            "not communicate this; a per-class recall of 0.000 on putdown does.")

    para(d, "4.2. Generalisable Observations on Pipeline Construction", H2)
    para(d, "Two of the three findings concern machinery surrounding the model rather than the "
            "model, and both proved more consequential than any architectural change.")
    para(d, "The interval-construction result is a specific instance of a general hazard: where the "
            "unit of prediction is coarser than the unit being predicted, the aggregation rule "
            "imposes a ceiling on the evaluation metric that is readily misattributed to the "
            "classifier. The ceiling d / w is simple enough to compute in advance, and we would "
            "encourage its explicit calculation whenever sliding-window detection is evaluated "
            "under temporal IoU. In the present case it accounted for the difference between an "
            "apparently failing system and a usable one.")
    para(d, "The crop-conditioning result is a second instance of the same class of problem. "
            "Conditioning information available only for positive examples will, through whatever "
            "fallback the implementation selects, encode the label. The fallback in our case "
            "appeared innocuous and would not have been visible in any loss curve. The remedy, "
            "extending conditioning across a padding interval so that positive and negative "
            "examples share geometry, generalises to any actor-conditioned or region-conditioned "
            "formulation.")
    para(d, "Both were identified by rendering the output of the data loader and inspecting it, "
            "which is why we treat visual inspection as a required verification step rather than an "
            "optional diagnostic convenience.")

    para(d, "4.3. Limitations", H2)
    para(d, "The most significant limitation concerns conditioning. Candidates and crop regions are "
            "derived from the ground-truth annotation boxes, and are used at inference as well as "
            "during training. The model is therefore supplied with the actor's spatial region and "
            "the approximate temporal neighbourhood of each event, neither of which is available at "
            "deployment. The reported figures should accordingly be read as an upper bound on what "
            "this route would achieve in production rather than as the performance of a deployable "
            "system, and substituting a pose-derived candidate and crop source is the principal "
            "outstanding item of work.")
    para(d, "The corpus comprises five recording days, so the validation and test splits each "
            "consist of a single day, and every reported figure for either split describes one "
            "day's shelves, lighting and shoppers over 32 and 35 events respectively. Decision "
            "thresholds selected on 32 events from a single day cannot be expected to transfer, and "
            "did not. We regard the resulting estimates as high in variance rather than incorrect: "
            "validation should be read as an optimistic bound and the held-out day as a single "
            "noisy sample. Grouped cross-validation over all five days, reporting mean and spread "
            "across folds, would yield a substantially more trustworthy figure at five times the "
            "computational cost, and its absence is the principal methodological limitation of this "
            "study. The held-out split has additionally been read twice, once per training regime; "
            "neither reading informed any decision, but the split is no longer perfectly naive.")
    para(d, "The corpus derives from a single camera in a single store, so nothing reported here "
            "establishes that the directional failure is a general property of video transformers "
            "rather than of this viewpoint. Actor identity is per-interaction rather than "
            "per-person, since CVAT tracks are drawn per event; two interactions by the same "
            "shopper form two actor streams. The synthetic negatives constructed for the seven "
            "cleared recordings are conditioned on borrowed boxes that do not correspond to any "
            "person's location, and are valid negatives for the transfer question but not for a "
            "presence question. Thirteen of the 261 annotated intervals remained in draft review "
            "status and were included; an option to exclude them exists but was not exercised. "
            "Ignore intervals are sparse, three across 42 recordings, so ignore handling is "
            "exercised but not stressed. Finally, the fine-tuning comparison uses a five-epoch "
            "budget with a warm-started head, and a longer schedule, stronger augmentation or "
            "minority-class oversampling might alter the balance between the two regimes.")

    para(d, "4.4. Future Work", H2)
    para(d, "The most direct response to the directional failure is to supervise direction "
            "explicitly rather than expecting it to be acquired incidentally. Three routes appear "
            "worthwhile, in decreasing order of expected value.")
    para(d, "Day-level grouped cross-validation should be undertaken first, because without it no "
            "subsequent comparison can be trusted at this data scale. Direction could then be made "
            "an explicit learning target: time-reversal augmentation, in which a reversed pickup is "
            "presented as a putdown, forces the representation to encode ordering, and a pretext "
            "task predicting frame order within a window or a loss term defined over the "
            "hand-state and shelf-state transition sequence would serve a comparable purpose. "
            "Third, direction may be supplied from outside the appearance model: a parallel "
            "component of the same system estimates hand state (empty or carrying) and shelf state "
            "(object removed, placed or unchanged) from frozen image embeddings, and the sign of "
            "the shelf-state transition is precisely the quantity the window classifier fails to "
            "transfer. Fusing that estimate as a directional prior, with the video transformer "
            "supplying detection and localisation and the state classifier supplying direction, "
            "matches each component to the sub-problem it demonstrably solves.")

    # ---------------- 5. Conclusions ----------------
    para(d, "5. Conclusions", H1)
    para(d, "We trained an actor-conditioned VideoMAE window classifier for retail pickup and "
            "putdown detection on 42 annotated recordings, and found that the pipeline surrounding "
            "the model mattered more than the model itself. Replacing span-derived with "
            "centre-derived interval boundaries raised validation F1 at tIoU 0.3 from 0.350 to "
            "0.647 without retraining, because a window longer than the event it detects imposes an "
            "arithmetic ceiling on temporal IoU. Holding annotation boxes across a padding interval "
            "removed a crop-geometry shortcut that would otherwise have permitted the classifier to "
            "separate the classes without observing the action.")
    para(d, "Partial fine-tuning of the final two transformer blocks improved validation event F1 "
            f"from 0.585 to {num(m['ft_val'], 'tiou@0.3', 'f1')} and held-out-day performance from "
            f"{num(m['fz_test'], 'tiou@0.3', 'f1')} to {num(m['ft_test'], 'tiou@0.3', 'f1')}. The "
            "gap between those figures is the substantive result. Detection and temporal "
            "localisation transfer across recording days; the pickup and putdown direction does "
            "not. On the held-out day true putdowns received a mean putdown probability of 0.062 "
            "against a pickup probability of 0.588, with no window exceeding 0.467, leaving the "
            "system an interaction detector with a pickup bias.")
    para(d, "We conclude that the direction of transfer requires explicit supervision rather than "
            "emerging from class labels alone, and that at five recording days grouped "
            "cross-validation over days should constitute the reporting standard. For deployment, "
            "per-class recall on the minority class rather than aggregate F1 is the quantity that "
            "should govern the decision.")

    # ---------------- Back matter ----------------
    rich(d, [("Author Contributions: ", "b"),
             "Conceptualization, F.L. and F.L.; methodology, F.L.; software, F.L.; validation, "
             "F.L. and F.L.; formal analysis, F.L.; investigation, F.L.; data curation, F.L.; "
             "writing—original draft preparation, F.L.; writing—review and editing, F.L.; "
             "visualization, F.L.; supervision, F.L. All authors have read and agreed to the "
             "published version of the manuscript."], BACK)
    rich(d, [("Funding: ", "b"), "This research received no external funding."], BACK)
    rich(d, [("Institutional Review Board Statement: ", "b"),
             "Not applicable. The study analyses pre-existing retail surveillance footage that was "
             "anonymised at source before being made available to the research team. No personally "
             "identifying features were accessible to the annotators or the authors."], BACK)
    rich(d, [("Informed Consent Statement: ", "b"),
             "Not applicable. No identifiable personal data were processed."], BACK)
    rich(d, [("Data Availability Statement: ", "b"),
             "The complete source code, configuration and evaluation harness are openly available "
             f"at {REPO_URL} (branch {BRANCH}). Derived annotation artefacts — canonical event "
             "tables, ignore intervals, actor tracks, window manifests, per-window model scores, "
             "trained model weights and all evaluation metrics — are provided as a reproducibility "
             "bundle with a SHA-256 manifest, available from the corresponding author on reasonable "
             "request. The source recordings are commercial retail footage of members of the public "
             "and are not publicly redistributable; the bundle carries each recording's storage "
             "location, size and SHA-256 digest so that the exact input set can be reconstructed "
             "and verified by holders of the appropriate credentials. All figures and tables in "
             "this manuscript are generated directly from the stored metrics files, and the "
             "generating scripts are included in the code release."], BACK)
    rich(d, [("Acknowledgments: ", "b"),
             "The authors thank the annotation team for the CVAT labelling effort underlying this "
             "study."], BACK)
    rich(d, [("Conflicts of Interest: ", "b"), "The authors declare no conflict of interest."], BACK)

    # ---------------- References ----------------
    para(d, "References", H1)
    for ref in [
        "Idrees, H.; Zamir, A.R.; Jiang, Y.-G.; Gorban, A.; Laptev, I.; Sukthankar, R.; Shah, M. "
        "The THUMOS challenge on action recognition for videos “in the wild”. Comput. Vis. Image "
        "Underst. 2017, 155, 1–23. https://doi.org/10.1016/j.cviu.2016.10.018",
        "Caba Heilbron, F.; Escorcia, V.; Ghanem, B.; Carlos Niebles, J. ActivityNet: A large-scale "
        "video benchmark for human activity understanding. In Proceedings of the IEEE Conference on "
        "Computer Vision and Pattern Recognition (CVPR), Boston, MA, USA, 7–12 June 2015; "
        "pp. 961–970.",
        "Lin, T.; Liu, X.; Li, X.; Ding, E.; Wen, S. BMN: Boundary-matching network for temporal "
        "action proposal generation. In Proceedings of the IEEE/CVF International Conference on "
        "Computer Vision (ICCV), Seoul, Korea, 27 October–2 November 2019; pp. 3889–3898. "
        "https://doi.org/10.48550/arXiv.1907.09702",
        "Zhang, C.-L.; Wu, J.; Li, Y. ActionFormer: Localizing moments of actions with transformers. "
        "In Computer Vision – ECCV 2022; Lecture Notes in Computer Science, Vol. 13664; Springer: "
        "Cham, Switzerland, 2022; pp. 492–510. https://doi.org/10.1007/978-3-031-19772-7_29",
        "Goyal, R.; Ebrahimi Kahou, S.; Michalski, V.; Materzynska, J.; Westphal, S.; Kim, H.; "
        "Haenel, V.; Fruend, I.; Yianilos, P.; Mueller-Freitag, M.; et al. The “something something” "
        "video database for learning and evaluating visual common sense. In Proceedings of the IEEE "
        "International Conference on Computer Vision (ICCV), Venice, Italy, 22–29 October 2017; "
        "pp. 5842–5850.",
        "Tong, Z.; Song, Y.; Wang, J.; Wang, L. VideoMAE: Masked autoencoders are data-efficient "
        "learners for self-supervised video pre-training. In Advances in Neural Information "
        "Processing Systems 35 (NeurIPS 2022); 2022. https://doi.org/10.48550/arXiv.2203.12602",
        "Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; "
        "Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. An image is worth 16×16 words: "
        "Transformers for image recognition at scale. In Proceedings of the International Conference "
        "on Learning Representations (ICLR), 2021. https://doi.org/10.48550/arXiv.2010.11929",
        "Kay, W.; Carreira, J.; Simonyan, K.; Zhang, B.; Hillier, C.; Vijayanarasimhan, S.; Viola, "
        "F.; Green, T.; Back, T.; Natsev, P.; et al. The Kinetics human action video dataset. arXiv "
        "2017, arXiv:1705.06950. https://doi.org/10.48550/arXiv.1705.06950",
        "CVAT.ai Corporation. Computer Vision Annotation Tool (CVAT), software, 2026. "
        "https://doi.org/10.5281/zenodo.3497105",
        "Zhang, Y.; Sun, P.; Jiang, Y.; Yu, D.; Weng, F.; Yuan, Z.; Luo, P.; Liu, W.; Wang, X. "
        "ByteTrack: Multi-object tracking by associating every detection box. In Computer Vision – "
        "ECCV 2022; Lecture Notes in Computer Science, Vol. 13682; Springer: Cham, Switzerland, "
        "2022; pp. 1–21. https://doi.org/10.1007/978-3-031-20047-2_1",
    ]:
        para(d, ref, REFS)

    output.parent.mkdir(parents=True, exist_ok=True)
    d.save(str(output))
    working = output.with_name("_template_working.docx")
    if working.exists():
        working.unlink()
    print(f"wrote {output} ({output.stat().st_size/1000:.0f} kB)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", type=Path,
                   default=Path.home() / "Downloads/jimaging-template.dot")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / ".local/paper/jimaging/jimaging_pickup_putdown.docx")
    p.add_argument("--figures", type=Path, default=REPO_ROOT / ".local/paper/figures")
    a = p.parse_args()
    build(a.template, a.output, a.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
