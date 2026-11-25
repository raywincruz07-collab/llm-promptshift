"""Tests for annotate_fresh.py: full round trip with programmatically filled sheets."""

import sys
from pathlib import Path

import numpy as np
import pytest
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "promptshift"))

import annotate_fresh as A  # noqa: E402
import config as C  # noqa: E402
from common import read_jsonl, write_jsonl  # noqa: E402


def fill(path, rule, leave_blank=0):
    wb = load_workbook(path)
    ws = wb["Label"]
    for i, row in enumerate(ws.iter_rows(min_row=2)):
        if i < leave_blank:
            continue
        row[2].value = rule(str(row[0].value), row[1].value)
    wb.save(path)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    fresh, ann, main = tmp_path / "fresh", tmp_path / "fresh" / "annotation", tmp_path / "main"
    monkeypatch.setattr(C, "stage_dir", lambda stage: main)
    write_jsonl(fresh / "sample_main.jsonl",
                [{"row_id": f"main{i:04d}", "text": f"post {i} {'damn' if i % 4 == 0 else 'fine'}",
                  "keyword": "you are", "window": "2026-05-01"} for i in range(400)])
    write_jsonl(fresh / "sample_calibration.jsonl",
                [{"row_id": f"calibration{i:04d}", "text": f"calib {i}"} for i in range(30)])
    write_jsonl(main / "sample.jsonl",
                [{"row_id": i, "label": int(i % 3 == 0), "text": f"olid {i} {'idiot' if i % 3 == 0 else 'ok'}"}
                 for i in range(860)])
    return fresh, ann, tmp_path


def test_round_trip(setup):
    fresh, ann, tmp = setup
    info = A.make_sheets(ann, fresh)
    assert info == {"calibration": 30, "main_items": 500, "fresh": 400, "anchors": 100}
    with pytest.raises(SystemExit):
        A.make_sheets(ann, fresh)  # never overwrite

    key = read_jsonl(ann / "key.jsonl")
    assert len({k["item_id"] for k in key}) == len(key) == 530
    order_a = [r[0].value for r in load_workbook(ann / "sheet_A.xlsx")["Label"].iter_rows(min_row=2)]
    order_b = [r[0].value for r in load_workbook(ann / "sheet_B.xlsx")["Label"].iter_rows(min_row=2)]
    assert sorted(order_a) == sorted(order_b) and order_a != order_b
    header = [c.value for c in load_workbook(ann / "sheet_A.xlsx")["Label"][1]]
    assert "olid" not in " ".join(map(str, header)).lower()  # no source column

    truth = {k["item_id"]: ("OFF" if ("damn" in k["text"] or "idiot" in k["text"]) else "NOT") for k in key}
    fill(ann / "calibration_A.xlsx", lambda i, t: truth[i])
    fill(ann / "calibration_B.xlsx", lambda i, t: truth[i])
    assert A.calib_compare(ann)["agree"] == 30

    fill(ann / "sheet_A.xlsx", lambda i, t: truth[i], leave_blank=3)
    with pytest.raises(SystemExit):
        A.score(ann)  # incomplete sheet is rejected
    fill(ann / "sheet_A.xlsx", lambda i, t: truth[i])
    flip = {k["item_id"] for k in key if k["source"] == "fresh"}
    flip = set(sorted(flip)[:20])
    fill(ann / "sheet_B.xlsx", lambda i, t: ("NOT" if truth[i] == "OFF" else "OFF") if i in flip else truth[i])
    r = A.score(ann)
    assert r["fresh"]["n_disagreements"] == 20
    assert r["olid_anchor"]["accuracy_vs_olid_A"] == 1.0
    assert 0.8 < r["fresh"]["krippendorff_alpha"] < 1.0

    with pytest.raises(SystemExit):
        A.finalize(ann, fresh, tmp)  # C not done yet
    fill(ann / "adjudication_C.xlsx", lambda i, t: truth[i])
    meta = A.finalize(ann, fresh, tmp)
    gold = read_jsonl(fresh / "gold_main.jsonl")
    assert meta["n"] == 400 and sum(not g["annotators_agree"] for g in gold) == 20
    assert all(g["label"] == int("damn" in g["text"]) for g in gold)  # majority restores truth
    assert (tmp / "annotation_fingerprint.json").exists()
