"""Human gold labels for the fresh posts (PLAN_v3.md, section 4.3).

Procedure (mirrors OLID: two annotations per post, a third on disagreement, majority label):
    python src/promptshift/annotate_fresh.py sheets          # calibration + main sheets (.xlsx)
    -> annotators A and B label calibration_A/B.xlsx (30 posts), then
    python src/promptshift/annotate_fresh.py calib-compare   # discuss disagreements, refine understanding
    -> A and B label sheet_A.xlsx / sheet_B.xlsx independently (500 posts: 400 fresh + 100 OLID anchors)
    python src/promptshift/annotate_fresh.py score           # agreement + adjudication sheet for C
    -> annotator C labels adjudication_C.xlsx (disagreements only, blind to A and B)
    python src/promptshift/annotate_fresh.py finalize        # gold_main.jsonl + fingerprint

The 100 OLID anchors are mixed in so we can measure how well our labels match OLID's.
Pilot posts are not labelled: the pilot only checks answer format.
Sheets contain post text: they stay under data/ (gitignored) and are shared only inside the team.

Code drafted with AI assistance (Claude, Anthropic); the team reviewed it and
is responsible for it (course AI-use policy).
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import stats as S  # noqa: E402
from common import git_state, read_json, read_jsonl, sha256_file, sha256_text, utc_now, write_json, write_jsonl  # noqa: E402
from prompts import OLID_NOT_OFFENSIVE, OLID_OFFENSIVE  # noqa: E402

FRESH_DIR = C.DATA_DIR / "fresh"
ANN_DIR = FRESH_DIR / "annotation"
SEED = 20261130
N_ANCHORS = 100
VALID = {"OFF": 1, "NOT": 0}

GUIDE = [
    "LABELLING GUIDE (read fully before starting)",
    "",
    "Task: for each post, choose OFF (offensive) or NOT (not offensive) in the 'label' column.",
    "",
    "Definitions (OLID, Zampieri et al. 2019, used verbatim):",
    f"  OFF: {OLID_OFFENSIVE}",
    f"  NOT: {OLID_NOT_OFFENSIVE}",
    "  The OLID paper adds that OFF includes insults, threats, and posts with profane language or swear words.",
    "",
    "Rules",
    "  1. Judge only the text shown. @USER replaces a username, URL replaces a link.",
    "  2. Any profanity or swear word -> OFF, even if it is friendly or not aimed at anyone.",
    "  3. An insult, threat or demeaning remark aimed at a person or group -> OFF, even if worded politely (veiled).",
    "  4. Strong disagreement or political criticism WITHOUT insult or profanity -> NOT.",
    "  5. If offensive words appear in the post (also when quoting someone) -> OFF.",
    "  6. Hashtags are part of the text.",
    "  7. Unsure? Pick the better label and put 'yes' in the 'unsure' column. Never leave 'label' empty.",
    "",
    "Invented examples (not from the data)",
    "  OFF  '@USER you are a complete idiot'                         (direct insult)",
    "  OFF  'this damn printer jammed again'                          (profanity, not targeted)",
    "  OFF  'people like her should crawl back under their rock'      (veiled insult)",
    "  NOT  'I disagree with the new gun control bill; it won't reduce crime'",
    "  NOT  'She is the best coach our team ever had'",
    "  NOT  'Conservatives and liberals should both read the actual bill URL'",
    "",
    "How to work",
    "  - Work alone. Do not discuss posts with the other annotator until both sheets are finished.",
    "  - Do not look posts up online and do not use any AI tool.",
    "  - Take a break every 100 posts. Some posts are hostile or upsetting: you may pause or stop at any time.",
    "  - Save the file under the same name when done.",
]


# ------------------------------------------------------------------ workbook I/O
def write_sheet(path, items, order_seed, title):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    rng = np.random.default_rng(order_seed)
    ordered = [items[i] for i in rng.permutation(len(items))]
    wb = Workbook()
    guide = wb.active
    guide.title = "Guide"
    guide.column_dimensions["A"].width = 120
    for line in [title, ""] + GUIDE:
        guide.append([line])
    guide["A1"].font = Font(bold=True, size=13)
    ws = wb.create_sheet("Label")
    ws.append(["item_id", "post", "label (OFF/NOT)", "unsure (yes/blank)", "comment"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for item in ordered:
        ws.append([item["item_id"], item["text"], None, None, None])
    widths = {"A": 10, "B": 100, "C": 16, "D": 16, "E": 30}
    for col, width in widths.items():
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2, min_col=2, max_col=2):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    label_dv = DataValidation(type="list", formula1='"OFF,NOT"', allow_blank=True)
    unsure_dv = DataValidation(type="list", formula1='"yes"', allow_blank=True)
    ws.add_data_validation(label_dv)
    ws.add_data_validation(unsure_dv)
    label_dv.add(f"C2:C{len(ordered) + 1}")
    unsure_dv.add(f"D2:D{len(ordered) + 1}")
    wb.active = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def read_sheet(path, expected_ids):
    """Return {item_id: (label_int, unsure_bool)}; stop with a clear message if incomplete."""
    from openpyxl import load_workbook

    if not Path(path).exists():
        raise SystemExit(f"{path} not found.")
    ws = load_workbook(path, read_only=True, data_only=True)["Label"]
    out, problems = {}, []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        item_id = str(row[0]).strip()
        label = str(row[2] or "").strip().upper()
        if label not in VALID:
            problems.append(item_id)
            continue
        out[item_id] = (VALID[label], str(row[3] or "").strip().lower() in {"yes", "y", "1", "x"})
    missing = sorted(set(expected_ids) - set(out))
    if problems or missing:
        shown = sorted(set(problems) | set(missing))
        raise SystemExit(f"{Path(path).name}: {len(shown)} posts without a valid label (OFF/NOT), "
                         f"e.g. {shown[:10]}. Fill them in and run again.")
    return out


# ------------------------------------------------------------------ steps
def build_items(fresh_dir=FRESH_DIR, seed=SEED):
    main = read_jsonl(Path(fresh_dir) / "sample_main.jsonl")
    calib = read_jsonl(Path(fresh_dir) / "sample_calibration.jsonl")
    olid = read_jsonl(C.stage_dir("main") / "sample.jsonl")
    if not main or not calib or not olid:
        raise SystemExit("need data/fresh/sample_main.jsonl, sample_calibration.jsonl and data/main/sample.jsonl")
    rng = np.random.default_rng([seed, 1])
    anchors = [olid[i] for i in sorted(rng.choice(len(olid), size=N_ANCHORS, replace=False))]
    codes = rng.permutation(10000)[: len(main) + len(anchors) + len(calib)]
    key, k = [], 0
    for source, records in (("fresh", main), ("olid_anchor", anchors), ("calibration", calib)):
        for r in records:
            key.append({"item_id": f"P{codes[k]:04d}", "source": source, "row_id": r["row_id"],
                        "olid_gold": int(r["label"]) if source == "olid_anchor" else None,
                        "text": r["text"]})
            k += 1
    return key


def make_sheets(ann_dir=ANN_DIR, fresh_dir=FRESH_DIR, seed=SEED):
    ann_dir = Path(ann_dir)
    if (ann_dir / "key.jsonl").exists():
        raise SystemExit("sheets already created; refusing to overwrite (they may contain labels).")
    key = build_items(fresh_dir, seed)
    write_jsonl(ann_dir / "key.jsonl", key)
    calib = [k for k in key if k["source"] == "calibration"]
    main = [k for k in key if k["source"] != "calibration"]
    write_sheet(ann_dir / "calibration_A.xlsx", calib, [seed, 2], "Calibration sheet, annotator A")
    write_sheet(ann_dir / "calibration_B.xlsx", calib, [seed, 3], "Calibration sheet, annotator B")
    write_sheet(ann_dir / "sheet_A.xlsx", main, [seed, 4], "Main sheet, annotator A")
    write_sheet(ann_dir / "sheet_B.xlsx", main, [seed, 5], "Main sheet, annotator B")
    return {"calibration": len(calib), "main_items": len(main),
            "fresh": sum(k["source"] == "fresh" for k in key), "anchors": N_ANCHORS}


def calib_compare(ann_dir=ANN_DIR):
    ann_dir = Path(ann_dir)
    key = [k for k in read_jsonl(ann_dir / "key.jsonl") if k["source"] == "calibration"]
    ids = [k["item_id"] for k in key]
    a, b = read_sheet(ann_dir / "calibration_A.xlsx", ids), read_sheet(ann_dir / "calibration_B.xlsx", ids)
    disagree = [k for k in key if a[k["item_id"]][0] != b[k["item_id"]][0]]
    alpha = S.krippendorff_alpha_nominal([[a[i][0], b[i][0]] for i in ids])
    print(f"Calibration: {len(ids) - len(disagree)}/{len(ids)} agree, Krippendorff alpha {alpha:.3f}")
    print("Discuss each disagreement against the guide rules (do not change the calibration sheets afterwards):")
    for k in disagree:
        print(f"  {k['item_id']}  A={'OFF' if a[k['item_id']][0] else 'NOT'}  "
              f"B={'OFF' if b[k['item_id']][0] else 'NOT'}  | {k['text']}")
    return {"n": len(ids), "agree": len(ids) - len(disagree), "alpha": alpha}


def kappa(x, y):
    x, y = np.asarray(x), np.asarray(y)
    po = float(np.mean(x == y))
    pe = float(np.mean(x) * np.mean(y) + (1 - np.mean(x)) * (1 - np.mean(y)))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(ann_dir=ANN_DIR):
    from openpyxl import Workbook  # noqa: F401  (fail early if missing)

    ann_dir = Path(ann_dir)
    key = [k for k in read_jsonl(ann_dir / "key.jsonl") if k["source"] != "calibration"]
    ids = [k["item_id"] for k in key]
    a, b = read_sheet(ann_dir / "sheet_A.xlsx", ids), read_sheet(ann_dir / "sheet_B.xlsx", ids)
    fresh = [k for k in key if k["source"] == "fresh"]
    anchors = [k for k in key if k["source"] == "olid_anchor"]
    result = {"created_utc": utc_now()}
    for name, group in (("fresh", fresh), ("olid_anchor", anchors)):
        ga = [a[k["item_id"]][0] for k in group]
        gb = [b[k["item_id"]][0] for k in group]
        result[name] = {"n": len(group), "raw_agreement": float(np.mean(np.equal(ga, gb))),
                        "krippendorff_alpha": S.krippendorff_alpha_nominal(np.column_stack([ga, gb])),
                        "offensive_share_A": float(np.mean(ga)), "offensive_share_B": float(np.mean(gb)),
                        "unsure_A": sum(a[k["item_id"]][1] for k in group),
                        "unsure_B": sum(b[k["item_id"]][1] for k in group)}
    gold = [k["olid_gold"] for k in anchors]
    for who, labels in (("A", a), ("B", b)):
        mine = [labels[k["item_id"]][0] for k in anchors]
        result["olid_anchor"][f"accuracy_vs_olid_{who}"] = float(np.mean(np.equal(mine, gold)))
        result["olid_anchor"][f"kappa_vs_olid_{who}"] = kappa(mine, gold)
    disagree = [k for k in fresh if a[k["item_id"]][0] != b[k["item_id"]][0]]
    result["fresh"]["n_disagreements"] = len(disagree)
    write_sheet(ann_dir / "adjudication_C.xlsx", disagree, [SEED, 6],
                "Adjudication sheet, annotator C (label each post yourself; you do not see A's or B's labels)")
    write_json(ann_dir / "scores.json", result)
    return result


def finalize(ann_dir=ANN_DIR, fresh_dir=FRESH_DIR, protocol_dir=C.PROTOCOL_DIR):
    ann_dir, fresh_dir = Path(ann_dir), Path(fresh_dir)
    key = [k for k in read_jsonl(ann_dir / "key.jsonl") if k["source"] != "calibration"]
    ids = [k["item_id"] for k in key]
    a, b = read_sheet(ann_dir / "sheet_A.xlsx", ids), read_sheet(ann_dir / "sheet_B.xlsx", ids)
    fresh = [k for k in key if k["source"] == "fresh"]
    disagree_ids = [k["item_id"] for k in fresh if a[k["item_id"]][0] != b[k["item_id"]][0]]
    c = read_sheet(ann_dir / "adjudication_C.xlsx", disagree_ids) if disagree_ids else {}
    sample = {r["row_id"]: r for r in read_jsonl(fresh_dir / "sample_main.jsonl")}
    gold = []
    for k in fresh:
        la, lb = a[k["item_id"]][0], b[k["item_id"]][0]
        final = la if la == lb else c[k["item_id"]][0]   # majority of three
        record = sample[k["row_id"]]
        gold.append({"row_id": record["row_id"], "text": record["text"], "label": final,
                     "label_A": la, "label_B": lb, "label_C": None if la == lb else c[k["item_id"]][0],
                     "annotators_agree": la == lb, "unsure_any": a[k["item_id"]][1] or b[k["item_id"]][1],
                     "keyword": record.get("keyword"), "window": record.get("window")})
    gold.sort(key=lambda r: r["row_id"])
    write_jsonl(fresh_dir / "gold_main.jsonl", gold)
    scores = read_json(ann_dir / "scores.json")
    meta = {"created_utc": utc_now(), "git": git_state(C.ROOT), "n": len(gold),
            "label_counts": dict(Counter(r["label"] for r in gold)),
            "offensive_share": float(np.mean([r["label"] for r in gold])),
            "n_disagreements_adjudicated": len(disagree_ids), "scores": scores,
            "gold_file_sha256": sha256_file(fresh_dir / "gold_main.jsonl"),
            "key_file_sha256": sha256_file(ann_dir / "key.jsonl"),
            "procedure": "two independent annotations; third annotator (blind) on disagreement; majority label"}
    write_json(fresh_dir / "gold_meta.json", meta)
    write_json(Path(protocol_dir) / "annotation_fingerprint.json",
               {k: v for k, v in meta.items()} | {"texts_sha256": sha256_text("".join(r["text"] for r in gold))})
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["sheets", "calib-compare", "score", "finalize"])
    args = parser.parse_args()
    if args.step == "sheets":
        print("SHEETS_OK", make_sheets(), "->", ANN_DIR)
    elif args.step == "calib-compare":
        calib_compare()
    elif args.step == "score":
        r = score()
        print("SCORE_OK", {"fresh_alpha": round(r["fresh"]["krippendorff_alpha"], 3),
                           "fresh_raw_agreement": round(r["fresh"]["raw_agreement"], 3),
                           "disagreements_for_C": r["fresh"]["n_disagreements"],
                           "anchor_accuracy_A": round(r["olid_anchor"]["accuracy_vs_olid_A"], 3),
                           "anchor_accuracy_B": round(r["olid_anchor"]["accuracy_vs_olid_B"], 3)})
    else:
        m = finalize()
        print("FINALIZE_OK", {k: m[k] for k in ("n", "label_counts", "offensive_share", "n_disagreements_adjudicated")})


if __name__ == "__main__":
    main()
