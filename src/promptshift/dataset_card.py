"""Describe and release the fresh Bluesky dataset (supervisor feedback, 27 Nov 2026).

    python src/promptshift/dataset_card.py describe   # distribution tables, no text -> artifacts/fresh/
    python src/promptshift/dataset_card.py release    # IDs + labels, side-by-side sheet -> data/fresh/release/
    python src/promptshift/dataset_card.py reasons    # after the team fills the 'reason' column
    python src/promptshift/dataset_card.py verify-ids # fetch all 400 posts by ID and check they match
    python src/promptshift/dataset_card.py verify-ids --sample 3   # live demo: 3 random posts

`describe` writes only counts (safe to commit). `release` writes post text and Bluesky IDs:
those files stay under data/ (gitignored) and go to the course submission, not to GitHub.

Code drafted with AI assistance (Claude, Anthropic); the team reviewed it and is responsible for it.
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
from annotate_fresh import GUIDE  # noqa: E402
from collect_fresh import normalise_post  # noqa: E402
from common import read_json, read_jsonl, utc_now, write_json  # noqa: E402

FRESH_DIR = C.DATA_DIR / "fresh"
ANN_DIR = FRESH_DIR / "annotation"
RELEASE_DIR = FRESH_DIR / "release"
OUT_DIR = C.ROOT / "artifacts" / "fresh"
SIDE_BY_SIDE = "annotations_side_by_side_CONTAINS_TWEET_TEXT.xlsx"
LAB = {1: "OFF", 0: "NOT", None: ""}
REASONS = [
    "profanity not aimed at anyone",
    "criticism vs insult (political or personal)",
    "veiled or implicit insult",
    "sarcasm or irony",
    "quoted or reported offensive words",
    "context missing (reply/link/image)",
    "slip or mistake",
    "other",
]
GUIDE_PROVENANCE = [
    "Provenance: the OFF/NOT definitions are OLID's (Zampieri et al. 2019), used verbatim.",
    "The rules and invented examples below were drafted with an LLM (Claude) and reviewed by the team.",
]


# ------------------------------------------------------------------ loading
def month(window):
    return (window or "")[:7]


def words(text):
    return len((text or "").split())


def bsky_url(uri):
    """at://did:plc:x/app.bsky.feed.post/rkey -> https://bsky.app/profile/did:plc:x/post/rkey"""
    if not uri or not uri.startswith("at://"):
        return ""
    parts = uri[len("at://"):].split("/")
    return f"https://bsky.app/profile/{parts[0]}/post/{parts[-1]}" if len(parts) == 3 else ""


def read_sheet_full(path):
    """{item_id: {"label": 'OFF'/'NOT', "unsure": bool, "comment": str}} from a filled labelling sheet."""
    from openpyxl import load_workbook

    if not Path(path).exists():
        return {}
    ws = load_workbook(path, read_only=True, data_only=True)["Label"]
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        row = list(row) + [None] * 5
        out[str(row[0]).strip()] = {"label": str(row[2] or "").strip().upper(),
                                    "unsure": str(row[3] or "").strip().lower() in {"yes", "y", "1", "x"},
                                    "comment": str(row[4] or "").strip()}
    return out


def load(fresh_dir=FRESH_DIR, ann_dir=ANN_DIR):
    fresh_dir, ann_dir = Path(fresh_dir), Path(ann_dir)
    gold = read_jsonl(fresh_dir / "gold_main.jsonl")
    if not gold:
        raise SystemExit(f"{fresh_dir / 'gold_main.jsonl'} not found: run annotate_fresh.py finalize first.")
    key = {k["row_id"]: k for k in read_jsonl(ann_dir / "key.jsonl") if k["source"] == "fresh"}
    sample = {r["row_id"]: r for r in read_jsonl(fresh_dir / "sample_main.jsonl")}
    sheets = {who: read_sheet_full(ann_dir / name) for who, name in
              (("A", "sheet_A.xlsx"), ("B", "sheet_B.xlsx"), ("C", "adjudication_C.xlsx"))}
    rows = []
    for g in gold:
        item = key.get(g["row_id"], {}).get("item_id", "")
        s = sample.get(g["row_id"], {})
        rows.append({
            "item_id": item, "row_id": g["row_id"], "keyword": g.get("keyword") or s.get("keyword"),
            "month": month(g.get("window") or s.get("window")), "created_at": s.get("created_at", ""),
            "author_hash": s.get("author_hash", ""), "text": g["text"],
            "A": g["label_A"], "B": g["label_B"], "C": g["label_C"], "gold": g["label"],
            "agree": g["annotators_agree"],
            "unsure_A": sheets["A"].get(item, {}).get("unsure", False),
            "unsure_B": sheets["B"].get(item, {}).get("unsure", False),
            "comment_A": sheets["A"].get(item, {}).get("comment", ""),
            "comment_B": sheets["B"].get(item, {}).get("comment", ""),
            "comment_C": sheets["C"].get(item, {}).get("comment", ""),
        })
    return rows


# ------------------------------------------------------------------ describe
def share(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def describe(rows, olid_rows=None, scores=None):
    n = len(rows)
    dis = [r for r in rows if not r["agree"]]
    d = {
        "created_utc": utc_now(), "n": n,
        "gold_offensive": sum(r["gold"] for r in rows), "gold_offensive_share": share(r["gold"] for r in rows),
        "annotator_A_offensive": sum(r["A"] for r in rows), "annotator_B_offensive": sum(r["B"] for r in rows),
        "both_offensive": sum(r["A"] == 1 and r["B"] == 1 for r in rows),
        "both_not": sum(r["A"] == 0 and r["B"] == 0 for r in rows),
        "A_off_B_not": sum(r["A"] == 1 and r["B"] == 0 for r in rows),
        "A_not_B_off": sum(r["A"] == 0 and r["B"] == 1 for r in rows),
        "disagreements": len(dis), "disagreement_rate": len(dis) / n if n else float("nan"),
        "C_labelled_offensive": sum(r["C"] == 1 for r in dis),
        "C_sided_with_A": sum(r["C"] == r["A"] for r in dis),
        "C_sided_with_B": sum(r["C"] == r["B"] for r in dis),
        "unsure_A": sum(r["unsure_A"] for r in rows), "unsure_B": sum(r["unsure_B"] for r in rows),
        "disagreements_flagged_unsure": sum(r["unsure_A"] or r["unsure_B"] for r in dis),
        "words_median_new": sorted(words(r["text"]) for r in rows)[n // 2] if n else None,
        "words_mean_new": share(words(r["text"]) for r in rows),
    }
    by_kw, by_month = defaultdict(list), defaultdict(list)
    for r in rows:
        by_kw[r["keyword"]].append(r)
        by_month[r["month"]].append(r)
    d["by_keyword"] = {k: {"n": len(v), "gold_offensive": sum(r["gold"] for r in v),
                           "gold_share": share(r["gold"] for r in v), "A_share": share(r["A"] for r in v),
                           "B_share": share(r["B"] for r in v), "disagreements": sum(not r["agree"] for r in v)}
                       for k, v in sorted(by_kw.items(), key=lambda kv: -share(r["gold"] for r in kv[1]))}
    d["by_month"] = {m: {"n": len(v), "gold_offensive": sum(r["gold"] for r in v),
                         "gold_share": share(r["gold"] for r in v), "disagreements": sum(not r["agree"] for r in v)}
                     for m, v in sorted(by_month.items())}
    if olid_rows:
        lab = [int(r["label"]) for r in olid_rows]
        lens = sorted(words(r["text"]) for r in olid_rows)
        d["olid_test"] = {"n": len(lab), "offensive": sum(lab), "offensive_share": share(lab),
                          "words_median": lens[len(lens) // 2], "words_mean": share(lens)}
    if scores:
        d["olid_anchors"] = scores.get("olid_anchor")
    return d


def describe_md(d):
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    lines = [
        "# Fresh Bluesky dataset: description (generated by dataset_card.py describe)", "",
        f"- Posts: {d['n']}; offensive (final label): {d['gold_offensive']} ({pct(d['gold_offensive_share'])})",
        f"- Annotator A offensive: {d['annotator_A_offensive']}; annotator B offensive: {d['annotator_B_offensive']}",
        f"- Agreement table: both OFF {d['both_offensive']}, both NOT {d['both_not']}, "
        f"A OFF / B NOT {d['A_off_B_not']}, A NOT / B OFF {d['A_not_B_off']}",
        f"- Disagreements: {d['disagreements']} ({pct(d['disagreement_rate'])}); annotator C labelled "
        f"{d['C_labelled_offensive']} of them OFF (sided with A {d['C_sided_with_A']}, with B {d['C_sided_with_B']})",
        f"- Marked 'unsure': A {d['unsure_A']}, B {d['unsure_B']}; disagreements with an 'unsure' flag: "
        f"{d['disagreements_flagged_unsure']}",
        f"- Words per post: median {d['words_median_new']}, mean {d['words_mean_new']:.1f}",
    ]
    if "olid_test" in d:
        o = d["olid_test"]
        lines.append(f"- OLID test for comparison: {o['n']} posts, {o['offensive']} offensive "
                     f"({pct(o['offensive_share'])}); words median {o['words_median']}, mean {o['words_mean']:.1f}")
    lines += ["", "## Offensive share by keyword", "",
              "| Keyword | Posts | Offensive | Share | A share | B share | Disagreements |", "|---|---|---|---|---|---|---|"]
    for k, v in d["by_keyword"].items():
        lines.append(f"| {k} | {v['n']} | {v['gold_offensive']} | {pct(v['gold_share'])} | {pct(v['A_share'])} | "
                     f"{pct(v['B_share'])} | {v['disagreements']} |")
    lines += ["", "## By month", "", "| Month | Posts | Offensive | Share | Disagreements |", "|---|---|---|---|---|"]
    for m, v in d["by_month"].items():
        lines.append(f"| {m} | {v['n']} | {v['gold_offensive']} | {pct(v['gold_share'])} | {v['disagreements']} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ release
def attach_uris(rows, raw_pool):
    index = defaultdict(list)
    for r in raw_pool:
        index[(r.get("text"), r.get("created_at"), r.get("author_hash"))].append(r.get("uri"))
    missing = ambiguous = 0
    for r in rows:
        uris = sorted(set(index.get((r["text"], r["created_at"], r["author_hash"]), [])))
        if not uris:
            missing += 1
        elif len(uris) > 1:
            ambiguous += 1
        r["uri"] = uris[0] if uris else ""
        r["url"] = bsky_url(r["uri"])
    return missing, ambiguous


ID_FIELDS = ["item_id", "row_id", "uri", "url", "keyword", "month", "created_at",
             "label_A", "label_B", "label_C", "label_final"]


def to_record(r, with_text=False):
    rec = {"item_id": r["item_id"], "row_id": r["row_id"], "uri": r.get("uri", ""), "url": r.get("url", ""),
           "keyword": r["keyword"], "month": r["month"], "created_at": r["created_at"],
           "label_A": LAB[r["A"]], "label_B": LAB[r["B"]], "label_C": LAB[r["C"]], "label_final": LAB[r["gold"]]}
    if with_text:
        rec["text"] = r["text"]
    return rec


def write_csv(path, records, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def write_side_by_side(path, rows, d):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    amber = PatternFill("solid", fgColor="FBEFD5")
    head = ["item_id", "row_id", "Bluesky link", "keyword", "month", "post", "A", "B", "C", "final",
            "A unsure", "B unsure", "A comment", "B comment", "C comment"]
    widths = [9, 10, 20, 14, 9, 80, 6, 6, 6, 7, 9, 9, 28, 28, 28]

    def fill(ws, subset, extra=()):
        ws.append(head + list(extra))
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in subset:
            ws.append([r["item_id"], r["row_id"], r.get("url", ""), r["keyword"], r["month"], r["text"],
                       LAB[r["A"]], LAB[r["B"]], LAB[r["C"]], LAB[r["gold"]],
                       "yes" if r["unsure_A"] else "", "yes" if r["unsure_B"] else "",
                       r["comment_A"], r["comment_B"], r["comment_C"]] + [""] * len(extra))
            if not r["agree"]:
                for c in ws[ws.max_row]:
                    c.fill = amber
        for i, wd in enumerate(widths + [34] * len(extra)):
            ws.column_dimensions[chr(65 + i)].width = wd
        for row in ws.iter_rows(min_row=2, min_col=6, max_col=6):
            row[0].alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "B2"
        ws.auto_filter.ref = ws.dimensions

    wb = Workbook()
    s = wb.active
    s.title = "Summary"
    s.column_dimensions["A"].width = 110
    for line in describe_md(d).splitlines():
        s.append([line])
    s["A1"].font = Font(bold=True, size=13)
    fill(wb.create_sheet("All posts"), rows)
    dis_ws = wb.create_sheet("Disagreements")
    fill(dis_ws, [r for r in rows if not r["agree"]], extra=("reason", "notes"))
    dv = DataValidation(type="list", formula1='"' + ",".join(REASONS) + '"', allow_blank=True)
    dis_ws.add_data_validation(dv)
    dv.add(f"P2:P{dis_ws.max_row}")
    g = wb.create_sheet("Guide")
    g.column_dimensions["A"].width = 120
    for line in GUIDE_PROVENANCE + [""] + GUIDE:
        g.append([line])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def release(fresh_dir=FRESH_DIR, ann_dir=ANN_DIR, out_dir=RELEASE_DIR):
    rows = load(fresh_dir, ann_dir)
    missing, ambiguous = attach_uris(rows, read_jsonl(Path(fresh_dir) / "raw_pool.jsonl"))
    d = describe(rows, read_jsonl(C.stage_dir("main") / "sample.jsonl"), read_json_safe(Path(ann_dir) / "scores.json"))
    out_dir = Path(out_dir)
    write_csv(out_dir / "bluesky2026_offensive_ids_labels.csv", [to_record(r) for r in rows], ID_FIELDS)
    write_csv(out_dir / "bluesky2026_offensive_CONTAINS_TWEET_TEXT.csv", [to_record(r, True) for r in rows],
              ID_FIELDS + ["text"])
    write_side_by_side(out_dir / SIDE_BY_SIDE, rows, d)
    return {"n": len(rows), "uris_found": len(rows) - missing, "uris_missing": missing,
            "uris_ambiguous": ambiguous, "out_dir": str(out_dir)}


def read_json_safe(path):
    return read_json(path) if Path(path).exists() else None


# ------------------------------------------------------------------ reasons
def reasons(path=RELEASE_DIR / SIDE_BY_SIDE, out_dir=OUT_DIR):
    from openpyxl import load_workbook

    ws = load_workbook(path, read_only=True, data_only=True)["Disagreements"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    counts = Counter(str(r[15] or "").strip() or "(not filled)" for r in rows if r and r[0])
    direction = Counter(f"A {r[6]} / B {r[7]} / C {r[8]}" for r in rows if r and r[0])
    result = {"created_utc": utc_now(), "n": sum(counts.values()), "reasons": dict(counts.most_common()),
              "label_patterns": dict(direction.most_common())}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(out_dir) / "disagreement_reasons.json", result)
    return result


# ------------------------------------------------------------------ verify-ids
APPVIEW = "https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts"
FULL_CSV = RELEASE_DIR / "bluesky2026_offensive_CONTAINS_TWEET_TEXT.csv"


def fetch_posts(uris, opener=None, batch=25, delay=0.3):
    """{uri: post view} from Bluesky's public API (no login needed). Missing = deleted or hidden."""
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    opener = opener or urlopen
    found = {}
    for i in range(0, len(uris), batch):
        query = urlencode([("uris", u) for u in uris[i:i + batch]])
        with opener(Request(f"{APPVIEW}?{query}", headers={"Accept": "application/json"}), timeout=60) as resp:
            for post in json.loads(resp.read().decode("utf-8")).get("posts") or []:
                found[post.get("uri")] = post
        if delay and i + batch < len(uris):
            time.sleep(delay)
    return found


def check_record(row, post):
    if post is None:
        return "not found (deleted or hidden)"
    record = post.get("record") or {}
    same_text = normalise_post(record.get("text") or "", record.get("facets")) == row["text"]
    same_time = (record.get("createdAt") or "") == row["created_at"]
    if same_text and same_time:
        return "match"
    return "text differs" if not same_text else "date differs"


def verify_ids(full_csv=FULL_CSV, out_dir=OUT_DIR, release_dir=RELEASE_DIR, opener=None, sample=0, seed=None):
    with open(full_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if sample:
        rows = random.Random(seed).sample(rows, min(sample, len(rows)))
    posts = fetch_posts([r["uri"] for r in rows], opener=opener)
    status = [(r, check_record(r, posts.get(r["uri"]))) for r in rows]
    if sample:
        for r, st in status:
            print(f"{r['item_id']}  {st:30s}  {r['url']}")
        return {"sample": len(status), "match": sum(st == "match" for _, st in status)}
    counts = {}
    for _, st in status:
        counts[st] = counts.get(st, 0) + 1
    result = {"checked_utc": utc_now(), "endpoint": APPVIEW, "n": len(status),
              "match": counts.get("match", 0), "status_counts": counts}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(out_dir) / "id_verification.json", result)
    write_csv(Path(release_dir) / "id_verification.csv",
              [{"item_id": r["item_id"], "url": r["url"], "status": st} for r, st in status],
              ["item_id", "url", "status"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["describe", "release", "reasons", "verify-ids"])
    parser.add_argument("--sample", type=int, default=0, help="verify-ids: only N random posts, printed")
    args = parser.parse_args()
    if args.step == "describe":
        rows = load()
        d = describe(rows, read_jsonl(C.stage_dir("main") / "sample.jsonl"), read_json_safe(ANN_DIR / "scores.json"))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(OUT_DIR / "dataset_description.json", d)
        (OUT_DIR / "dataset_description.md").write_text(describe_md(d), encoding="utf-8")
        print(describe_md(d))
        print("DESCRIBE_OK ->", OUT_DIR / "dataset_description.md")
    elif args.step == "release":
        print("RELEASE_OK", release())
    elif args.step == "reasons":
        print("REASONS_OK", reasons())
    else:
        print("VERIFY_IDS_OK", verify_ids(sample=args.sample))


if __name__ == "__main__":
    main()
