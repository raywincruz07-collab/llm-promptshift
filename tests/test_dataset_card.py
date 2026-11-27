"""Synthetic checks for dataset_card.py (no real data, no network)."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "promptshift"))

import dataset_card as D  # noqa: E402
from annotate_fresh import write_sheet  # noqa: E402


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _fixture(tmp_path):
    fresh, ann = tmp_path / "fresh", tmp_path / "fresh" / "annotation"
    kws = ["you are", "MAGA"]
    # A, B labels: 2 both OFF, 4 both NOT, 2 A OFF / B NOT (C: OFF, NOT)
    labels = [(1, 1, None), (1, 1, None), (0, 0, None), (0, 0, None), (0, 0, None), (0, 0, None),
              (1, 0, 1), (1, 0, 0)]
    sample, gold, key, raw = [], [], [], []
    for i, (a, b, c) in enumerate(labels):
        rid = f"main{i:04d}"
        text = f"post number {i} with some words"
        rec = {"row_id": rid, "text": text, "keyword": kws[i % 2], "window": "2026-0%d-01" % (5 + i % 2),
               "created_at": f"2026-05-0{i + 1}T10:00:00Z", "author_hash": f"h{i}"}
        sample.append(rec)
        raw.append({"uri": f"at://did:plc:u{i}/app.bsky.feed.post/r{i}", "text": text,
                    "created_at": rec["created_at"], "author_hash": rec["author_hash"]})
        final = a if a == b else c
        gold.append({"row_id": rid, "text": text, "label": final, "label_A": a, "label_B": b,
                     "label_C": c, "annotators_agree": a == b, "keyword": rec["keyword"], "window": rec["window"]})
        key.append({"item_id": f"P{i:04d}", "source": "fresh", "row_id": rid, "olid_gold": None, "text": text})
    _jsonl(fresh / "sample_main.jsonl", sample)
    _jsonl(fresh / "gold_main.jsonl", gold)
    _jsonl(fresh / "raw_pool.jsonl", raw)
    _jsonl(ann / "key.jsonl", key)
    return fresh, ann


def test_describe_counts(tmp_path):
    fresh, ann = _fixture(tmp_path)
    rows = D.load(fresh, ann)
    d = D.describe(rows, olid_rows=[{"label": 1, "text": "a b c"}, {"label": 0, "text": "a b"}])
    assert d["n"] == 8 and d["gold_offensive"] == 3
    assert d["A_off_B_not"] == 2 and d["A_not_B_off"] == 0 and d["disagreements"] == 2
    assert d["C_labelled_offensive"] == 1 and d["C_sided_with_A"] == 1 and d["C_sided_with_B"] == 1
    assert sum(v["n"] for v in d["by_keyword"].values()) == 8
    assert d["olid_test"]["offensive"] == 1
    assert "Offensive share by keyword" in D.describe_md(d)


def test_release_files(tmp_path):
    fresh, ann = _fixture(tmp_path)
    out = tmp_path / "release"
    r = D.release(fresh, ann, out)
    assert r["uris_found"] == 8 and r["uris_missing"] == 0
    with open(out / "bluesky2026_offensive_ids_labels.csv", encoding="utf-8") as f:
        recs = list(csv.DictReader(f))
    assert "text" not in recs[0] and recs[0]["url"].startswith("https://bsky.app/profile/")
    assert (out / D.SIDE_BY_SIDE).exists()
    res = D.reasons(out / D.SIDE_BY_SIDE, tmp_path / "artifacts")
    assert res["n"] == 2 and res["reasons"] == {"(not filled)": 2}


def test_sheet_comments_are_read(tmp_path):
    fresh, ann = _fixture(tmp_path)
    items = [{"item_id": f"P{i:04d}", "text": "x"} for i in range(8)]
    write_sheet(ann / "sheet_A.xlsx", items, [1, 2], "A")
    from openpyxl import load_workbook
    wb = load_workbook(ann / "sheet_A.xlsx")
    ws = wb["Label"]
    for row in ws.iter_rows(min_row=2):
        row[2].value, row[4].value = "OFF", "because"
    wb.save(ann / "sheet_A.xlsx")
    rows = D.load(fresh, ann)
    assert all(r["comment_A"] == "because" for r in rows)


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_verify_ids_with_fake_api(tmp_path):
    from urllib.parse import parse_qs, urlparse

    fresh, ann = _fixture(tmp_path)
    out = tmp_path / "release"
    D.release(fresh, ann, out)
    raw = {r["uri"]: r for r in (json.loads(l) for l in (fresh / "raw_pool.jsonl").read_text().splitlines())}
    deleted = "at://did:plc:u0/app.bsky.feed.post/r0"

    def opener(request, timeout=60):
        uris = parse_qs(urlparse(request.full_url).query)["uris"]
        posts = [{"uri": u, "record": {"text": raw[u]["text"], "createdAt": raw[u]["created_at"]}}
                 for u in uris if u != deleted]
        if posts:
            posts[-1]["record"]["text"] += " edited"
        return _Resp({"posts": posts})

    res = D.verify_ids(out / "bluesky2026_offensive_CONTAINS_TWEET_TEXT.csv", tmp_path / "art", out, opener=opener)
    assert res["n"] == 8 and res["match"] == 6
    assert res["status_counts"]["not found (deleted or hidden)"] == 1
    assert res["status_counts"]["text differs"] == 1
    assert (out / "id_verification.csv").exists()
