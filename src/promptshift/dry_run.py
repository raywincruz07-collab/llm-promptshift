"""End-to-end dry run on synthetic posts with the mock API. No network, no real data.

    python src/promptshift/dry_run.py

Plants known effects (wording flips, repeat noise, a definition effect for
Gemma and Qwen only, memorisation for one model in the probe), runs every
step of the study in a temporary folder, and checks the analysis finds them.
Prints DRY_RUN_OK if everything works.
"""

import argparse
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
from analyze import analyse  # noqa: E402
from annotate import export as export_annotation, score as score_annotation  # noqa: E402
import csv  # noqa: E402
from common import read_json  # noqa: E402
from mock_api import MockClient  # noqa: E402
from prepare import prepare  # noqa: E402
from probe import analyse_probe, prepare_probe  # noqa: E402
from run import run  # noqa: E402

WORDS = ("this game was wild cant believe the ref missed that call again honestly "
         "people need to calm down about politics tonight great show love it").split()


def synthetic_posts(n, seed=7):
    rng = random.Random(seed)
    posts = []
    for i in range(n):
        length = rng.randint(6, 24)
        tokens = [rng.choice(WORDS) for _ in range(length)]
        if rng.random() < 0.5:
            tokens.insert(0, "@USER")
        if rng.random() < 0.2:
            tokens.append("URL")
        posts.append({"row_id": i, "label": int(rng.random() < 0.345),
                      "text": f"{' '.join(tokens)} #{i}"})
    return posts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", type=int, default=1012)
    parser.add_argument("--keep", type=Path, help="keep outputs in this folder")
    args = parser.parse_args()

    work = args.keep or Path(tempfile.mkdtemp(prefix="promptshift_dry_run_"))
    data, protocol = work / "data", work / "protocol"
    posts = synthetic_posts(args.posts)
    truth = {p["text"]: p["label"] for p in posts}

    meta = prepare(posts, "main", data, protocol)
    assert meta["n_requests"] == len(posts) * 3 * 8
    client = MockClient(
        truth=truth, seed=11,
        definition_recall_gain={"gemma4-26b": 0.15, "qwen3.6-36b": 0.15},
        definition_fp_change={"gemma4-26b": 0.03, "qwen3.6-36b": 0.03},
        transport_failure=0.01, noncompliance=0.002,
    )
    first = run(data / "requests.jsonl", data / "predictions.jsonl", data / "attempts.jsonl",
                client, limit=len(posts), delay=0, backoff_base=0,
                expected_manifest_sha256=meta["requests_file_sha256"], quiet=True)
    assert first["remaining"] > 0  # simulate an interrupted session
    final = run(data / "requests.jsonl", data / "predictions.jsonl", data / "attempts.jsonl",
                client, delay=0, backoff_base=0,
                expected_manifest_sha256=meta["requests_file_sha256"], quiet=True)
    assert final["remaining"] == 0, final

    results = analyse(data)
    by_test = {t["test"]: t for t in results["primary"]}
    for name in ("H1_Gemma", "H1_Qwen", "H1_Llama"):
        assert by_test[name]["significant_holm"], name
    if C.PRIMARY_H2 == "macro_f1":
        assert by_test["H2_Gemma"]["estimate"] > 0 and by_test["H2_Qwen"]["estimate"] > 0
        assert not by_test["H2_Llama"]["significant_holm"]

    probe_meta = prepare_probe(posts, "main", data, protocol, n_posts=min(200, len(posts)))
    items = {}
    for line in (data / "probe_items.jsonl").read_text().splitlines():
        item = __import__("json").loads(line)
        items[item["prefix"]] = item["reference"]
    probe_client = MockClient(memory=items, memorise_rate={"llama-3.3-70b": 0.5}, seed=5)
    run(data / "probe_requests.jsonl", data / "probe_predictions.jsonl", data / "probe_attempts.jsonl",
        probe_client, delay=0, backoff_base=0,
        expected_manifest_sha256=read_json(data / "probe_meta.json")["requests_file_sha256"], quiet=True)
    probe_rows = {r["model"]: r for r in analyse_probe(data)}
    assert probe_rows["llama-3.3-70b"]["ci_low"] > 0
    assert probe_rows["gemma4-26b"]["ci_low"] <= 0.05

    ann_dir = data / "annotation"
    exported = export_annotation(data, ann_dir)
    key = {k["item_id"]: k for k in __import__("json").loads(
        (ann_dir / "annotation_key_DO_NOT_SHOW_ANNOTATORS.json").read_text())["items"]}
    gold = {p["row_id"]: p["label"] for p in posts}
    rng = random.Random(3)
    for annotator in ("A", "B"):
        path = ann_dir / f"annotation_sheet_{annotator}_CONTAINS_TWEET_TEXT.csv"
        rows = list(csv.reader(path.open(encoding="utf-8")))
        for row in rows[2:]:
            item = key[int(row[0])]
            noise = 0.3 if item["group"] == "flip" else 0.05
            label = gold[item["row_id"]] if rng.random() > noise else 1 - gold[item["row_id"]]
            row[2] = "OFF" if label else "NOT"
        csv.writer(path.open("w", newline="", encoding="utf-8")).writerows(rows)
    ann_rows, ann_diff = score_annotation(ann_dir, data)
    assert ann_diff["ci_low"] > 0

    print("Primary tests on synthetic data (planted: H1 > 0 for all; H2 > 0 for Gemma and Qwen only):")
    for t in results["primary"]:
        print(f"  {t['test']:<10} {t['estimate']:+.4f}  95% CI [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]"
              f"  Holm p={t['p_holm']:.4f}")
    print("Probe (planted: memorisation for Llama only):")
    for model, r in probe_rows.items():
        print(f"  {C.MODEL_SHORT[model]:<6} guided − general ROUGE-L {r['mean_difference_guided_minus_general']:+.3f}"
              f"  [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
    print(f"Eligible probe posts: {probe_meta['eligible_posts']}")
    print(f"Human validation (planted: more disagreement on flip posts): flip − stable "
          f"{ann_diff['estimate']:+.3f} [{ann_diff['ci_low']:+.3f}, {ann_diff['ci_high']:+.3f}], exported {exported['exported']}")
    print("Outputs:", work)
    print("DRY_RUN_OK")


if __name__ == "__main__":
    main()
