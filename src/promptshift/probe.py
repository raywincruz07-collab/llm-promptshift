"""Contamination probe (guided vs general completion, after Golchin & Surdeanu 2024).

    python src/promptshift/probe.py prepare      # after prepare.py --stage main
    python src/promptshift/run.py --stage main --all --probe
    python src/promptshift/probe.py analyze

Each model gets the first half of a post twice: once with a general
"finish this post" instruction and once with a guided instruction naming
the dataset and split. If guided completions match the real second half
much better than general ones, that is evidence of memorisation. A null
result means "no evidence under this probe", not "no contamination".

TEAM CHECK before freezing: compare both templates with the paper's
guided/general instructions and record any differences in the protocol.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
from common import (  # noqa: E402
    read_json, read_jsonl, sha256_file, sha256_text, utc_now, write_json, write_jsonl,
)

GENERAL_TEMPLATE = (
    "Finish the second part of the following social-media post so that both parts "
    "together form a single post with the label below. Output only the second part.\n\n"
    "Label: {label}\nFirst part: {prefix}\nContinuation:"
)
GUIDED_TEMPLATE = (
    "You are given the first part of a post from the {split} split of the TweetEval "
    "offensive-language dataset (OLID). Finish the second part exactly as it appears in "
    "the dataset, relying only on the original form of the post in the dataset. "
    "Output only the second part.\n\n"
    "Label: {label}\nFirst part: {prefix}\nContinuation:"
)
SPLIT_NAME = {"main": "test", "pilot": "validation"}
LABEL_NAME = {0: "NOT_OFFENSIVE", 1: "OFFENSIVE"}


# -------------------------------------------------------------- preparation
def content_tokens(tokens):
    return [t for t in tokens if t not in C.PROBE_PLACEHOLDER_TOKENS]


def split_post(text):
    """Return (prefix, reference, n_content) or None if not eligible."""
    tokens = text.split()
    n_content = len(content_tokens(tokens))
    if n_content < C.PROBE_MIN_CONTENT_TOKENS:
        return None
    wanted = n_content // 2
    seen = 0
    for cut, token in enumerate(tokens, start=1):
        if token not in C.PROBE_PLACEHOLDER_TOKENS:
            seen += 1
        if seen == wanted:
            break
    return " ".join(tokens[:cut]), " ".join(tokens[cut:]), n_content


def prepare_probe(sample, sample_name, data_dir, protocol_dir,
                  seed=C.PROBE_SEED, n_posts=C.PROBE_N_POSTS):
    items = []
    for record in sample:
        split = split_post(record["text"])
        if split:
            prefix, reference, n_content = split
            items.append({"row_id": record["row_id"], "label": record["label"],
                          "prefix": prefix, "reference": reference, "n_content_tokens": n_content})
    eligible = len(items)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(eligible, size=min(n_posts, eligible), replace=False))
    items = [items[i] for i in chosen]

    requests = []
    for model in C.MODELS:
        for item in items:
            for kind, template in (("general", GENERAL_TEMPLATE), ("guided", GUIDED_TEMPLATE)):
                prompt = template.format(split=SPLIT_NAME[sample_name],
                                         label=LABEL_NAME[item["label"]], prefix=item["prefix"])
                requests.append({
                    "request_id": f"probe:{model}:{item['row_id']}:{kind}",
                    "task": "complete", "sample": sample_name, "model": model,
                    "row_id": item["row_id"], "condition": kind,
                    "max_tokens": C.MAX_TOKENS_PROBE,
                    "prompt_text": prompt, "prompt_sha256": sha256_text(prompt),
                })
    order = np.random.default_rng([seed, 1]).permutation(len(requests))
    requests = [requests[i] for i in order]
    for index, request in enumerate(requests):
        request["order"] = index

    data_dir, protocol_dir = Path(data_dir), Path(protocol_dir)
    write_jsonl(data_dir / "probe_items.jsonl", items)
    write_jsonl(data_dir / "probe_requests.jsonl", requests)
    meta = {
        "created_utc": utc_now(), "sample": sample_name, "seed": seed,
        "eligible_posts": eligible, "selected_posts": len(items),
        "min_content_tokens": C.PROBE_MIN_CONTENT_TOKENS,
        "n_requests": len(requests),
        "selected_row_ids_sha256": sha256_text(",".join(str(i["row_id"]) for i in items)),
        "requests_file_sha256": sha256_file(data_dir / "probe_requests.jsonl"),
        "general_template_sha256": sha256_text(GENERAL_TEMPLATE),
        "guided_template_sha256": sha256_text(GUIDED_TEMPLATE),
    }
    write_json(data_dir / "probe_meta.json", meta)
    write_json(protocol_dir / "probe_fingerprint.json", meta)
    return meta


# ----------------------------------------------------------------- scoring
def normalise(text):
    text = (text or "").lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return [t for t in text.split() if t not in {"user", "url"}]


def rouge_l_f1(candidate, reference):
    a, b = normalise(candidate), normalise(reference)
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def clean_completion(text):
    text = (text or "").strip()
    text = re.sub(r"^(continuation|second part)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"').strip()


def analyse_probe(data_dir, seed=C.BOOTSTRAP_SEED, n_boot=C.N_BOOTSTRAP):
    from scipy.stats import binomtest, wilcoxon

    data_dir = Path(data_dir)
    out = data_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    items = {i["row_id"]: i for i in read_jsonl(data_dir / "probe_items.jsonl")}
    requests = read_jsonl(data_dir / "probe_requests.jsonl")
    predictions = {p["request_id"]: p for p in read_jsonl(data_dir / "probe_predictions.jsonl")}
    if len(predictions) < len(requests):
        raise SystemExit(f"{len(requests) - len(predictions)} probe requests have no result yet.")

    rows, review = [], []
    for k, model in enumerate(C.MODELS):
        scores = {"general": {}, "guided": {}}
        exact = {"general": 0, "guided": 0}
        invalid = {"general": 0, "guided": 0}
        for row_id, item in items.items():
            for kind in ("general", "guided"):
                record = predictions[f"probe:{model}:{row_id}:{kind}"]
                if not record["compliant"]:
                    invalid[kind] += 1
                    continue
                completion = clean_completion(record["raw_response"])
                score = rouge_l_f1(completion, item["reference"])
                scores[kind][row_id] = score
                exact[kind] += normalise(completion) == normalise(item["reference"])
                if score >= C.PROBE_REVIEW_ROUGE_L:
                    review.append({"model": model, "row_id": row_id, "kind": kind,
                                   "rouge_l": round(score, 3), "prefix": item["prefix"],
                                   "reference": item["reference"], "completion": completion})
        paired = sorted(set(scores["general"]) & set(scores["guided"]))
        diffs = np.array([scores["guided"][r] - scores["general"][r] for r in paired])
        rng = np.random.default_rng([seed, 9, k])
        boot = diffs[rng.integers(0, len(diffs), size=(n_boot, len(diffs)))].mean(axis=1)
        nonzero = diffs[diffs != 0]
        sign = binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5, alternative="greater") if len(nonzero) else None
        wil = wilcoxon(diffs, alternative="greater") if len(nonzero) else None
        rows.append({
            "model": model, "n_items": len(items), "n_paired": len(paired),
            "invalid_general": invalid["general"], "invalid_guided": invalid["guided"],
            "mean_rouge_l_general": float(np.mean([scores["general"][r] for r in paired])),
            "mean_rouge_l_guided": float(np.mean([scores["guided"][r] for r in paired])),
            "mean_difference_guided_minus_general": float(diffs.mean()),
            "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)),
            "sign_test_p_greater": float(sign.pvalue) if sign else None,
            "wilcoxon_p_greater": float(wil.pvalue) if wil else None,
            "exact_matches_general": exact["general"], "exact_matches_guided": exact["guided"],
            f"rouge_l_ge_{C.PROBE_REVIEW_ROUGE_L}_general": sum(s >= C.PROBE_REVIEW_ROUGE_L for s in scores["general"].values()),
            f"rouge_l_ge_{C.PROBE_REVIEW_ROUGE_L}_guided": sum(s >= C.PROBE_REVIEW_ROUGE_L for s in scores["guided"].values()),
        })

    _write_csv(out / "probe_results.csv", rows)
    _write_csv(out / "probe_review_CONTAINS_TWEET_TEXT.csv", review)
    lines = ["# Contamination probe (generated tables, no interpretation)", "",
             "Limited behavioural probe for memorisation; not proof of exposure or of its absence.", "",
             "| Model | Paired n | ROUGE-L general | ROUGE-L guided | Guided − general [95% CI] | Sign test p | Wilcoxon p | Exact matches (gen/guided) |",
             "|---|---:|---:|---:|---|---:|---:|---|"]
    for r in rows:
        lines.append(
            f"| {C.MODEL_SHORT[r['model']]} | {r['n_paired']} | {r['mean_rouge_l_general']:.3f} | "
            f"{r['mean_rouge_l_guided']:.3f} | {r['mean_difference_guided_minus_general']:+.3f} "
            f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {_p(r['sign_test_p_greater'])} | "
            f"{_p(r['wilcoxon_p_greater'])} | {r['exact_matches_general']}/{r['exact_matches_guided']} |")
    lines += ["", f"{len(review)} completions with ROUGE-L ≥ {C.PROBE_REVIEW_ROUGE_L} listed for manual review "
              "in probe_review_CONTAINS_TWEET_TEXT.csv (gitignored)."]
    (out / "probe_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _p(value):
    return "n/a" if value is None else f"{value:.4f}"


def _write_csv(path, rows):
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["prepare", "analyze"])
    args = parser.parse_args()
    args.data_dir = C.stage_dir("main")
    if args.step == "prepare":
        sample = read_jsonl(args.data_dir / "sample.jsonl")
        meta = prepare_probe(sample, "main", args.data_dir, C.PROTOCOL_DIR)
        print("PROBE_PREPARE_OK", {k: meta[k] for k in ("eligible_posts", "selected_posts", "n_requests")})
    else:
        analyse_probe(args.data_dir)
        print("PROBE_ANALYSIS_OK ->", args.data_dir / "results" / "probe_summary.md")


if __name__ == "__main__":
    main()
