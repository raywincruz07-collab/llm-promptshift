"""Human validation: do models flip on posts that humans also find ambiguous?

    python src/promptshift/annotate.py export   # after analyze.py --stage main
    (two team members fill in their sheets independently)
    python src/promptshift/annotate.py score

export samples 50 posts where at least one model gave different labels across
the six W/D conditions ("flip") and 50 posts where every model was stable,
shuffles them, and writes one sheet per annotator WITHOUT the group, the gold
label or any model output (blind). Annotators label each post OFF or NOT
using the OLID definitions printed in the sheet header.

score computes human-human agreement (Krippendorff's alpha, raw agreement)
overall and per group, and the difference in human disagreement between flip
and stable posts with a bootstrap CI. Exploratory: if humans disagree more
on flip posts, model instability tracks genuine ambiguity.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import stats as S  # noqa: E402
from common import read_jsonl, write_json  # noqa: E402
from prompts import OLID_NOT_OFFENSIVE, OLID_OFFENSIVE  # noqa: E402

GROUP_CONDITIONS = ["W0", "W1", "W2", "D0", "D1", "D2"]
VALID = {"OFF": 1, "NOT": 0}


def flip_groups(sample, predictions):
    labels = {}
    for p in predictions:
        if p["compliant"] and p["condition"] in GROUP_CONDITIONS:
            labels.setdefault(p["row_id"], {}).setdefault(p["model"], set()).add(p["parsed_label"])
    flip, stable = [], []
    for record in sample:
        per_model = labels.get(record["row_id"], {})
        if any(len(v) > 1 for v in per_model.values()):
            flip.append(record["row_id"])
        elif len(per_model) == len(C.MODELS):
            stable.append(record["row_id"])
    return flip, stable


def export(data_dir, out_dir, seed=C.ANNOTATION_SEED):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    sample = read_jsonl(data_dir / "sample.jsonl")
    text = {r["row_id"]: r["text"] for r in sample}
    flip, stable = flip_groups(sample, read_jsonl(data_dir / "predictions.jsonl"))
    rng = np.random.default_rng(seed)
    pick_flip = list(rng.choice(flip, size=min(C.ANNOTATION_N_FLIP, len(flip)), replace=False))
    pick_stable = list(rng.choice(stable, size=min(C.ANNOTATION_N_STABLE, len(stable)), replace=False))
    items = [(int(r), "flip") for r in pick_flip] + [(int(r), "stable") for r in pick_stable]
    items = [items[i] for i in rng.permutation(len(items))]

    out_dir.mkdir(parents=True, exist_ok=True)
    key = [{"item_id": i + 1, "row_id": row_id, "group": group} for i, (row_id, group) in enumerate(items)]
    write_json(out_dir / "annotation_key_DO_NOT_SHOW_ANNOTATORS.json",
               {"n_flip_available": len(flip), "n_stable_available": len(stable), "items": key})
    header = (f"Label each post OFF or NOT. OFF: {OLID_OFFENSIVE} NOT: {OLID_NOT_OFFENSIVE} "
              "Work alone; do not discuss until both sheets are complete.")
    for annotator in ("A", "B"):
        path = out_dir / f"annotation_sheet_{annotator}_CONTAINS_TWEET_TEXT.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["instructions", header])
            writer.writerow(["item_id", "text", "label"])
            for item in key:
                writer.writerow([item["item_id"], text[item["row_id"]], ""])
    return {"flip_available": len(flip), "stable_available": len(stable), "exported": len(key)}


def read_sheet(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))[2:]
    labels = {}
    for item_id, _, label in rows:
        label = label.strip().upper()
        if label not in VALID:
            raise SystemExit(f"{path}: item {item_id} has label {label!r}; use OFF or NOT")
        labels[int(item_id)] = VALID[label]
    return labels


def score(out_dir, data_dir, seed=C.BOOTSTRAP_SEED):
    out_dir = Path(out_dir)
    key = read_jsonl_key(out_dir)
    gold = {r["row_id"]: r["label"] for r in read_jsonl(Path(data_dir) / "sample.jsonl")}
    a = read_sheet(out_dir / "annotation_sheet_A_CONTAINS_TWEET_TEXT.csv")
    b = read_sheet(out_dir / "annotation_sheet_B_CONTAINS_TWEET_TEXT.csv")
    rows = []
    disagree = {}
    for group in ("all", "flip", "stable"):
        items = [k for k in key if group == "all" or k["group"] == group]
        la = np.array([a[k["item_id"]] for k in items])
        lb = np.array([b[k["item_id"]] for k in items])
        g = np.array([gold[k["row_id"]] for k in items])
        disagree[group] = (la != lb).astype(float)
        rows.append({
            "group": group, "n": len(items),
            "krippendorff_alpha_A_B": S.krippendorff_alpha_nominal(np.column_stack([la, lb])),
            "raw_agreement_A_B": float(np.mean(la == lb)),
            "agreement_A_gold": float(np.mean(la == g)),
            "agreement_B_gold": float(np.mean(lb == g)),
        })
    rng = np.random.default_rng([seed, 11])
    f, s = disagree["flip"], disagree["stable"]
    boot = (f[rng.integers(0, len(f), (C.N_BOOTSTRAP, len(f)))].mean(1)
            - s[rng.integers(0, len(s), (C.N_BOOTSTRAP, len(s)))].mean(1))
    diff = S.summarise(f.mean() - s.mean(), boot)
    lines = ["# Human validation (generated, no interpretation)", "",
             "| Group | n | Krippendorff alpha (A,B) | Raw agreement | A vs gold | B vs gold |",
             "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['group']} | {r['n']} | {r['krippendorff_alpha_A_B']:.3f} | {r['raw_agreement_A_B']:.3f} | "
              f"{r['agreement_A_gold']:.3f} | {r['agreement_B_gold']:.3f} |" for r in rows]
    lines += ["", f"Human disagreement, flip minus stable: {diff['estimate']:+.3f} "
              f"(95% CI {diff['ci_low']:+.3f} to {diff['ci_high']:+.3f})."]
    (out_dir / "annotation_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_dir / "annotation_results.json", {"groups": rows, "flip_minus_stable_disagreement": diff})
    return rows, diff


def read_jsonl_key(out_dir):
    import json
    return json.loads((Path(out_dir) / "annotation_key_DO_NOT_SHOW_ANNOTATORS.json").read_text())["items"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["export", "score"])
    args = parser.parse_args()
    data_dir = C.stage_dir("main")
    out_dir = data_dir / "annotation"
    if args.step == "export":
        print("ANNOTATION_EXPORT_OK", export(data_dir, out_dir), "->", out_dir)
    else:
        rows, diff = score(out_dir, data_dir)
        print("ANNOTATION_SCORE_OK ->", out_dir / "annotation_results.md")


if __name__ == "__main__":
    main()
