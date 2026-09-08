"""Tests for scripts/convert_cvat_source_export.py (CVAT job-export mode)."""

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import convert_cvat_source_export as cvt  # noqa: E402

JOB_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <id>1</id>
      <name>clipA</name>
      <size>600</size>
      <assignee><username>ana@example.com</username></assignee>
    </task>
    <dumped>2026-09-07T18:35:00+00:00</dumped>
  </meta>
  <track id="0" label="pickup" source="manual">
    <box frame="100" keyframe="1" outside="0" occluded="0">
      <attribute name="notes"></attribute>
      <attribute name="confidence">high</attribute>
      <attribute name="hard_case">false</attribute>
      <attribute name="item_count">2</attribute>
      <attribute name="review_status">accepted</attribute>
    </box>
    <box frame="101" keyframe="0" outside="0" occluded="0">
      <attribute name="notes"></attribute>
      <attribute name="confidence">high</attribute>
      <attribute name="hard_case">false</attribute>
      <attribute name="item_count">2</attribute>
      <attribute name="review_status">accepted</attribute>
    </box>
  </track>
  <track id="1" label="ignore" source="manual">
    <box frame="500" keyframe="1" outside="0" occluded="0">
      <attribute name="ignore_reason">occluded</attribute>
      <attribute name="notes"></attribute>
      <attribute name="review_status">accepted</attribute>
    </box>
  </track>
</annotations>
"""

ZERO_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <id>3</id>
      <name>clipB</name>
      <size>300</size>
    </task>
    <dumped>2026-09-07T18:35:00+00:00</dumped>
  </meta>
</annotations>
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def workdir(tmp_path: Path) -> dict[str, Path]:
    unzipped = tmp_path / "unzipped"
    _write(unzipped / "clipA__task_1__job_2" / "annotations.xml", JOB_XML)
    _write(unzipped / "clipB__task_3__job_4" / "annotations.xml", ZERO_XML)

    _write(
        tmp_path / "export_manifest.csv",
        "task_id,job_id,source_video_filename,job_status,event_count,s3_uri,export_timestamp\n"
        "1,2,clipA.mp4,completed,2,s3://b/raw/clipA__task_1__job_2.zip,2026-09-07T18:35:00+00:00\n"
        "3,4,clipB.mp4,completed,0,s3://b/raw/clipB__task_3__job_4.zip,2026-09-07T18:35:00+00:00\n",
    )
    _write(
        tmp_path / "clips.csv",
        "clip_id,s3_key,duration_s,fps,width,height,n_person_tracks,usable,active_start_s,active_end_s,split,session_id,notes\n"
        "clipA,source_videos/clipA.mp4,20.0,30.0,1920,1080,1,True,0.0,20.0,,,\n"
        "clipB,source_videos/clipB.mp4,10.0,30.0,1920,1080,1,True,0.0,10.0,,,\n",
    )

    cands = tmp_path / "cands"
    _write(
        cands / "clipA" / "clipA.json",
        json.dumps(
            {
                "source_video_id": "clipA",
                "candidates": [
                    {
                        "candidate_id": "c1",
                        "source_start_s": 2.0,
                        "source_end_s": 5.0,
                        "actor_id": "a1",
                        "hand_side": "left",
                        "region_id": "r1",
                    },
                    {
                        "candidate_id": "c2",
                        "source_start_s": 10.0,
                        "source_end_s": 15.0,
                        "actor_id": "a1",
                        "hand_side": "left",
                        "region_id": "r1",
                    },
                ],
            }
        ),
    )
    _write(
        cands / "clipB" / "clipB.json",
        json.dumps(
            {
                "source_video_id": "clipB",
                "candidates": [
                    {
                        "candidate_id": "c3",
                        "source_start_s": 0.0,
                        "source_end_s": 4.0,
                        "actor_id": "a2",
                        "hand_side": "right",
                        "region_id": "r2",
                    },
                ],
            }
        ),
    )
    return {
        "unzipped": unzipped,
        "manifest": tmp_path / "export_manifest.csv",
        "clips": tmp_path / "clips.csv",
        "cands": cands,
        "out": tmp_path / "out",
    }


def test_parse_job_export() -> None:
    root = ET.fromstring(JOB_XML)
    anns = cvt.parse_cvat_1_1_job_export(root, fps=30.0)
    assert len(anns) == 2
    pickup = next(a for a in anns if a["label"] == "pickup")
    assert pickup["clip_id"] == "clipA"
    assert pickup["start_s"] == pytest.approx(100 / 30)
    assert pickup["end_s"] == pytest.approx(102 / 30)
    assert pickup["attributes"]["item_count"] == "2"
    assert pickup["annotator"] == "ana@example.com"
    ignore = next(a for a in anns if a["label"] == "ignore")
    assert ignore["attributes"]["ignore_reason"] == "occluded"


def test_build_candidate_review_mapping() -> None:
    anns = cvt.parse_cvat_1_1_job_export(ET.fromstring(JOB_XML), fps=30.0)
    windows = {
        "clipA": [
            {"candidate_id": "c1", "source_start_s": 2.0, "source_end_s": 5.0},
            {"candidate_id": "c2", "source_start_s": 10.0, "source_end_s": 15.0},
        ],
        "clipB": [{"candidate_id": "c3", "source_start_s": 0.0, "source_end_s": 4.0}],
    }
    rows, jsons, unlabeled = cvt.build_candidate_review(
        anns, windows, {"clipB"}, {}, Path("/tmp/never"), ".local/source_videos"
    )
    by_id = {r["candidate_id"]: r for r in rows}
    assert set(by_id) == {"c1", "c3"}  # c2: no overlap, clip has events -> unlabeled
    assert by_id["c1"]["review_groups"] == "cvat_event"
    assert by_id["c1"]["event_count"] == 1
    assert by_id["c3"]["review_groups"] == "cvat_zero_event"
    assert by_id["c3"]["event_count"] == 0
    assert unlabeled == 1
    ev = jsons["c1"]["events"][0]
    assert ev["label"] == "pickup"
    assert ev["start_s"] == round(100 / 30 - 2.0, 3)
    assert ev["end_s"] == round(102 / 30 - 2.0, 3)
    assert jsons["c3"]["events"] == []


def test_run_dir_mode_end_to_end(workdir: dict[str, Path], monkeypatch) -> None:
    argv = [
        "convert_cvat_source_export.py",
        "--cvat-export-dir",
        str(workdir["unzipped"]),
        "--clips-csv",
        str(workdir["clips"]),
        "--manifest",
        str(workdir["manifest"]),
        "--candidate-meta-dir",
        str(workdir["cands"]),
        "--output-dir",
        str(workdir["out"]),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cvt.main()

    out = workdir["out"]
    events = list(csv.DictReader((out / "events.csv").open()))
    assert len(events) == 2  # item_count=2 expands the pickup track
    assert {e["type"] for e in events} == {"pickup"}
    assert all(e["annotator"] == "ana@example.com" for e in events)

    import pyarrow.parquet as pq

    ignores = pq.read_table(out / "ignore_intervals.parquet").to_pylist()
    assert len(ignores) == 1
    assert ignores[0]["reason"] == "occluded"

    prov = pq.read_table(out / "event_provenance.parquet").to_pylist()
    assert len(prov) == 2
    assert all(p["review_status"] == "accepted" for p in prov)

    clips = list(csv.DictReader((out / "clips.csv").open()))
    by_clip = {c["clip_id"]: c for c in clips}
    assert by_clip["clipB"]["verified_negative"] == "True"
    assert by_clip["clipA"]["verified_negative"] == "False"
    assert by_clip["clipA"]["event_count"] == "2"

    rows = list(csv.DictReader((out / "review_manifest.csv").open()))
    assert {r["candidate_id"] for r in rows} == {"c1", "c3"}
    for r in rows:
        assert Path(r["json_path"]).exists()
        data = json.loads(Path(r["json_path"]).read_text())
        assert data["candidate_id"] == r["candidate_id"]


def test_dir_mode_manifest_mismatch_fails(workdir: dict[str, Path], monkeypatch, capsys) -> None:
    bad = workdir["manifest"].with_name("bad_manifest.csv")
    bad.write_text(
        "task_id,job_id,source_video_filename,job_status,event_count,s3_uri,export_timestamp\n"
        "1,2,clipA.mp4,completed,9,s3://b/raw/clipA__task_1__job_2.zip,2026-09-07T18:35:00+00:00\n",
        encoding="utf-8",
    )
    argv = [
        "convert_cvat_source_export.py",
        "--cvat-export-dir",
        str(workdir["unzipped"]),
        "--clips-csv",
        str(workdir["clips"]),
        "--manifest",
        str(bad),
        "--candidate-meta-dir",
        str(workdir["cands"]),
        "--output-dir",
        str(workdir["out"]),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        cvt.main()
    assert not workdir["out"].exists()
