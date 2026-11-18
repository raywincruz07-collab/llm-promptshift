"""Analysis: primary tests, secondary and sensitivity analyses, baselines, figures.

    python src/promptshift/analyze.py --stage pilot   # compliance check only
    python src/promptshift/analyze.py --stage main    # all pre-registered analyses

Refuses to run until every request has a result (no peeking at partial
results). Writes tables, figures and a numbers-only summary to
data/<stage>/results/.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import stats as S  # noqa: E402
from common import read_json, read_jsonl, sha256_file, utc_now, write_json  # noqa: E402

W = ["W0", "W1", "W2"]
D = ["D0", "D1", "D2"]


# ------------------------------------------------------------------ loading
def load(data_dir, allow_incomplete=False):
    data_dir = Path(data_dir)
    sample = read_jsonl(data_dir / "sample.jsonl")
    requests = read_jsonl(data_dir / "requests.jsonl")
    predictions = read_jsonl(data_dir / "predictions.jsonl")
    missing = len(requests) - len(predictions)
    if missing and not allow_incomplete:
        raise SystemExit(f"{missing} requests still have no result. Finish the run first.")
    row_ids = [r["row_id"] for r in sample]
    position = {row_id: i for i, row_id in enumerate(row_ids)}
    y = np.array([r["label"] for r in sample], dtype=int)
    labels = {
        model: {c: np.full(len(sample), np.nan) for c in C.CONDITIONS}
        for model in C.MODELS
    }
    for record in predictions:
        value = C.LABELS.get(record.get("parsed_label")) if record["compliant"] else None
        if value is not None:
            labels[record["model"]][record["condition"]][position[record["row_id"]]] = value
    return sample, y, labels, predictions, requests


def as_int(column):
    return column.astype(int)


# ----------------------------------------------------------- per-condition
def compliance_table(predictions, requests):
    expected = Counter((r["model"], r["condition"]) for r in requests)
    received = Counter((p["model"], p["condition"]) for p in predictions)
    compliant = Counter((p["model"], p["condition"]) for p in predictions if p["compliant"])
    rows = []
    for model in C.MODELS:
        for condition in C.CONDITIONS:
            key = (model, condition)
            rows.append({
                "model": model, "condition": condition,
                "requested": expected[key], "received": received[key],
                "compliant": compliant[key],
                "compliance_rate": round(compliant[key] / expected[key], 4) if expected[key] else None,
            })
    return rows


def condition_metrics(y, labels, rng):
    rows = []
    for model in C.MODELS:
        for condition in C.CONDITIONS:
            column = labels[model][condition]
            valid = ~np.isnan(column)
            yv, pv = y[valid], as_int(column[valid])
            idx = S.stratified_indices(yv, C.N_BOOTSTRAP, rng)
            row = {"model": model, "condition": condition, "n_valid": int(valid.sum())}
            for name, function in S.METRICS.items():
                row[name] = float(function(yv, pv))
            boot_f1 = S.macro_f1(yv[idx], pv[idx])
            boot_prev = S.prevalence(pv[idx])
            row["prevalence"] = float(S.prevalence(pv))
            row["macro_f1_ci_low"], row["macro_f1_ci_high"] = map(float, np.quantile(boot_f1, [0.025, 0.975]))
            row["prevalence_ci_low"], row["prevalence_ci_high"] = map(float, np.quantile(boot_prev, [0.025, 0.975]))
            rows.append(row)
    return rows


# ------------------------------------------------------------ primary tests
def complete_rows(labels, model, conditions=C.CONDITIONS):
    return np.all([~np.isnan(labels[model][c]) for c in conditions], axis=0)


def h1(y, labels, model, idx_rng, invalid_as_disagreement=False, subset=None):
    cols = {c: labels[model][c] for c in W + ["R0"]}
    if invalid_as_disagreement:
        keep = np.ones(len(y), bool) if subset is None else subset
        filled = {c: np.where(np.isnan(v), -1 - i, v) for i, (c, v) in enumerate(cols.items())}
        cols = {c: v[keep] for c, v in filled.items()}
        yk = y[keep]
    else:
        keep = complete_rows(labels, model)
        if subset is not None:
            keep &= subset
        cols = {c: v[keep] for c, v in cols.items()}
        yk = y[keep]
    x = S.mean_pairwise_disagreement([cols[c] for c in W]) - (cols["W0"] != cols["R0"])
    idx = S.stratified_indices(yk, C.N_BOOTSTRAP, idx_rng)
    result = S.summarise(x.mean(), x[idx].mean(axis=1), alternative="greater")
    result["n_posts"] = int(len(yk))
    return result


def h3(y, labels, model, metric, alternative, idx_rng, invalid_as_error=False, subset=None):
    function = S.METRICS[metric]
    if invalid_as_error:
        keep = np.ones(len(y), bool) if subset is None else subset
        cols = {c: np.where(np.isnan(labels[model][c]), 1 - y, labels[model][c])[keep].astype(int)
                for c in W + D}
    else:
        keep = complete_rows(labels, model)
        if subset is not None:
            keep &= subset
        cols = {c: as_int(labels[model][c][keep]) for c in W + D}
    yk = y[keep]
    observed = np.mean([function(yk, cols[c]) for c in D]) - np.mean([function(yk, cols[c]) for c in W])
    idx = S.stratified_indices(yk, C.N_BOOTSTRAP, idx_rng)
    yb = yk[idx]
    boot = (np.mean([function(yb, cols[c][idx]) for c in D], axis=0)
            - np.mean([function(yb, cols[c][idx]) for c in W], axis=0))
    result = S.summarise(observed, boot, alternative=alternative)
    result["n_posts"] = int(len(yk))
    return result


def h3_design():
    if C.PRIMARY_H2 == "macro_f1":
        return "macro_f1", list(C.MODELS), "two-sided"
    if C.PRIMARY_H2 == "offensive_recall":
        return "offensive_recall", ["gemma4-26b", "qwen3.6-36b"], "greater"
    raise ValueError(C.PRIMARY_H2)


def primary_family(y, labels, seed, **options):
    metric, h3_models, alternative = h3_design()
    tests = []
    for k, model in enumerate(C.MODELS):
        result = h1(y, labels, model, np.random.default_rng([seed, 1, k]),
                    invalid_as_disagreement=options.get("invalid_as_error", False),
                    subset=options.get("subset"))
        tests.append({"test": f"H1_{C.MODEL_SHORT[model]}", "model": model,
                      "estimand": "wording minus repeat disagreement", "sesoi": C.SESOI["H1"], **result})
    for k, model in enumerate(h3_models):
        result = h3(y, labels, model, metric, alternative, np.random.default_rng([seed, 3, k]),
                    invalid_as_error=options.get("invalid_as_error", False),
                    subset=options.get("subset"))
        tests.append({"test": f"H2_{C.MODEL_SHORT[model]}", "model": model,
                      "estimand": f"definition minus no definition: {metric}",
                      "sesoi": C.SESOI[metric], **result})
    adjusted = S.holm([t["p_value"] for t in tests])
    for test, p_holm in zip(tests, adjusted):
        test["p_holm"] = float(p_holm)
        test["significant_holm"] = bool(p_holm < C.ALPHA)
        test["size_verdict"] = S.size_verdict(test, test["sesoi"])
    return tests


# ---------------------------------------------------------------- secondary
def secondary(y, labels, seed):
    rows = []
    for k, model in enumerate(C.MODELS):
        keep = complete_rows(labels, model)
        cols = {c: labels[model][c][keep] for c in C.CONDITIONS}
        yk = y[keep]
        idx = S.stratified_indices(yk, C.N_BOOTSTRAP, np.random.default_rng([seed, 5, k]))

        def add(name, per_post, alternative="two-sided"):
            result = S.summarise(per_post.mean(), per_post[idx].mean(axis=1), alternative)
            rows.append({"model": model, "measure": name, "n_posts": int(len(yk)), **result})

        wording_w = S.mean_pairwise_disagreement([cols[c] for c in W])
        wording_d = S.mean_pairwise_disagreement([cols[c] for c in D])
        repeat_w = (cols["W0"] != cols["R0"]).astype(float)
        repeat_d = (cols["D0"] != cols["RD0"]).astype(float)
        add("wording disagreement, no definition (mean pairwise W0-W2)", wording_w)
        add("wording disagreement, definition (mean pairwise D0-D2)", wording_d)
        add("repeat disagreement W0 vs R0", repeat_w)
        add("repeat disagreement D0 vs RD0", repeat_d)
        add("H3: (wording-repeat with definition) minus (without)",
            (wording_d - repeat_d) - (wording_w - repeat_w))
        add("legacy any-flip rate across W0-W2",
            (np.ptp(np.vstack([cols[c] for c in W]), axis=0) > 0).astype(float))
        add("prevalence change: definition minus none (mean over wordings)",
            np.mean([cols[c] for c in D], axis=0) - np.mean([cols[c] for c in W], axis=0))

        for metric in ("offensive_recall", "not_offensive_recall", "macro_f1", "accuracy"):
            function = S.METRICS[metric]
            ints = {c: as_int(cols[c]) for c in W + D}
            observed = (np.mean([function(yk, ints[c]) for c in D])
                        - np.mean([function(yk, ints[c]) for c in W]))
            boot = (np.mean([function(yk[idx], ints[c][idx]) for c in D], axis=0)
                    - np.mean([function(yk[idx], ints[c][idx]) for c in W], axis=0))
            rows.append({"model": model, "measure": f"{metric}: definition minus none",
                         "n_posts": int(len(yk)), **S.summarise(observed, boot)})
        for group, conds in (("no definition", W), ("definition", D)):
            rows.append({"model": model, "measure": f"Krippendorff alpha across wordings, {group}",
                         "n_posts": int(len(yk)),
                         "estimate": S.krippendorff_alpha_nominal(np.column_stack([cols[c] for c in conds]))})

    for a, b in [(0, 1), (0, 2), (1, 2)]:
        ma, mb = C.MODELS[a], C.MODELS[b]
        keep = ~np.isnan(labels[ma]["W0"]) & ~np.isnan(labels[mb]["W0"])
        per_post = (labels[ma]["W0"][keep] != labels[mb]["W0"][keep]).astype(float)
        idx = S.stratified_indices(y[keep], C.N_BOOTSTRAP, np.random.default_rng([seed, 7, a, b]))
        rows.append({"model": f"{C.MODEL_SHORT[ma]} vs {C.MODEL_SHORT[mb]}",
                     "measure": "between-model disagreement on W0", "n_posts": int(keep.sum()),
                     **S.summarise(per_post.mean(), per_post[idx].mean(axis=1))})
    return rows


def baselines(y):
    """Majority class, and a random guesser matching the true class shares (expected values)."""
    p = float(np.mean(y == 1))
    zeros = np.zeros_like(y)
    return [
        {"baseline": "majority (always NOT_OFFENSIVE)",
         "accuracy": float(S.accuracy(y, zeros)), "macro_f1": float(S.macro_f1(y, zeros)),
         "offensive_recall": 0.0, "not_offensive_recall": 1.0, "prevalence": 0.0},
        {"baseline": "random, weighted by class shares (expected)",
         "accuracy": p * p + (1 - p) ** 2, "macro_f1": 0.5,
         "offensive_recall": p, "not_offensive_recall": 1 - p, "prevalence": p},
    ]


def pilot_report(data_dir):
    data_dir = Path(data_dir)
    requests = read_jsonl(data_dir / "requests.jsonl")
    predictions = read_jsonl(data_dir / "predictions.jsonl")
    if len(predictions) < len(requests):
        raise SystemExit(f"{len(requests) - len(predictions)} pilot requests have no result yet.")
    rows = compliance_table(predictions, requests)
    out = data_dir / "results"
    write_csv(out / "compliance.csv", rows)
    bad = [p["raw_response"] for p in predictions if not p["compliant"]]
    lines = ["# Pilot: compliance check only (no hypothesis is tested on the pilot)", "",
             f"Requests {len(requests)}, compliant {len(predictions) - len(bad)}.", "",
             "| Model | Condition | Compliant / requested |", "|---|---|---|"]
    lines += [f"| {C.MODEL_SHORT[r['model']]} | {r['condition']} | {r['compliant']}/{r['requested']} |" for r in rows]
    if bad:
        lines += ["", "Examples of non-compliant replies:", ""] + [f"- `{b!r}`" for b in bad[:10]]
    (out / "pilot_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows, bad


def singleton_mask(sample):
    counts = Counter(r["text"].strip() for r in sample)
    return np.array([counts[r["text"].strip()] == 1 for r in sample])


# ------------------------------------------------------------------ output
def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    return "" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def summary_markdown(meta, compliance, metrics, primary, secondary_rows, sensitivity, base_rows):
    lines = [
        "# Results (generated tables, no interpretation)",
        "",
        f"Generated {meta['generated_utc']} from predictions sha256 `{meta['predictions_sha256'][:16]}…`.",
        f"Sample `{meta['sample']}`, {meta['n_posts']} posts; bootstrap B={C.N_BOOTSTRAP}, seed {C.BOOTSTRAP_SEED}.",
        f"Primary H2 metric: `{C.PRIMARY_H2}`. Holm correction across {len(primary)} primary tests, alpha {C.ALPHA}.",
        "",
        "## Compliance",
        "",
        f"Total requests {sum(r['requested'] for r in compliance)}, compliant {sum(r['compliant'] for r in compliance)}.",
        "",
        "## Primary tests",
        "",
        "| Test | Estimate | 95% CI | One-sided 95% bound | p | p (Holm) | Significant | Size verdict | n |",
        "|---|---:|---|---:|---:|---:|---|---|---:|",
    ]
    for t in primary:
        bound = t["one_sided_lower"] if t["alternative"] == "greater" else None
        lines.append(
            f"| {t['test']} | {t['estimate']:+.4f} | [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] | "
            f"{fmt(bound)} | {t['p_value']:.4f} | {t['p_holm']:.4f} | {t['significant_holm']} | "
            f"{t['size_verdict']} | {t['n_posts']} |"
        )
    lines += ["", "## Baselines (no model)", "",
              "| Baseline | Accuracy | Macro-F1 | Off. recall | Not-off. recall | Predicted prevalence |",
              "|---|---:|---:|---:|---:|---:|"]
    for b in base_rows:
        lines.append(f"| {b['baseline']} | {b['accuracy']:.4f} | {b['macro_f1']:.4f} | "
                     f"{b['offensive_recall']:.4f} | {b['not_offensive_recall']:.4f} | {b['prevalence']:.4f} |")
    lines += ["", "## Metrics by condition", "",
              "| Model | Condition | n | Accuracy | Macro-F1 [95% CI] | Off. recall | Not-off. recall | Predicted prevalence [95% CI] |",
              "|---|---|---:|---:|---|---:|---:|---|"]
    for m in metrics:
        lines.append(
            f"| {C.MODEL_SHORT[m['model']]} | {m['condition']} | {m['n_valid']} | {m['accuracy']:.4f} | "
            f"{m['macro_f1']:.4f} [{m['macro_f1_ci_low']:.4f}, {m['macro_f1_ci_high']:.4f}] | "
            f"{m['offensive_recall']:.4f} | {m['not_offensive_recall']:.4f} | "
            f"{m['prevalence']:.4f} [{m['prevalence_ci_low']:.4f}, {m['prevalence_ci_high']:.4f}] |"
        )
    lines += ["", "## Secondary measures", "", "| Model | Measure | Estimate | 95% CI | n |", "|---|---|---:|---|---:|"]
    for s in secondary_rows:
        ci = f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]" if "ci_low" in s else ""
        lines.append(f"| {C.MODEL_SHORT.get(s['model'], s['model'])} | {s['measure']} | {s['estimate']:+.4f} | {ci} | {s['n_posts']} |")
    lines += ["", "## Sensitivity analyses (primary estimands)", "",
              "| Analysis | Test | Estimate | 95% CI | p (Holm within analysis) | n |", "|---|---|---:|---|---:|---:|"]
    for name, tests in sensitivity.items():
        for t in tests:
            lines.append(f"| {name} | {t['test']} | {t['estimate']:+.4f} | [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] | {t['p_holm']:.4f} | {t['n_posts']} |")
    return "\n".join(lines) + "\n"


def analyse(data_dir, allow_incomplete=False, make_figures=True):
    data_dir = Path(data_dir)
    out = data_dir / "results"
    sample, y, labels, predictions, requests = load(data_dir, allow_incomplete)
    sample_name = requests[0]["sample"] if requests else None

    compliance = compliance_table(predictions, requests)
    metrics = condition_metrics(y, labels, np.random.default_rng([C.BOOTSTRAP_SEED, 0]))
    primary = primary_family(y, labels, C.BOOTSTRAP_SEED)
    secondary_rows = secondary(y, labels, C.BOOTSTRAP_SEED)
    sensitivity = {
        "invalid answers counted as errors": primary_family(y, labels, C.BOOTSTRAP_SEED, invalid_as_error=True),
        "exact-duplicate posts removed": primary_family(y, labels, C.BOOTSTRAP_SEED, subset=singleton_mask(sample)),
    }

    write_csv(out / "compliance.csv", compliance)
    write_csv(out / "metrics_by_condition.csv", metrics)
    write_csv(out / "primary_tests.csv", primary)
    write_csv(out / "secondary.csv", secondary_rows)
    write_csv(out / "sensitivity.csv", [{"analysis": k, **t} for k, tests in sensitivity.items() for t in tests])
    write_csv(out / "baselines.csv", baselines(y))

    meta = {
        "generated_utc": utc_now(),
        "sample": sample_name,
        "n_posts": len(sample),
        "predictions_sha256": sha256_file(data_dir / "predictions.jsonl"),
        "manifest_meta": read_json(data_dir / "manifest_meta.json"),
        "primary_h2": C.PRIMARY_H2,
        "n_bootstrap": C.N_BOOTSTRAP,
        "bootstrap_seed": C.BOOTSTRAP_SEED,
        "complete": len(predictions) == len(requests),
    }
    write_json(out / "analysis_meta.json", meta)
    (out / "summary.md").write_text(
        summary_markdown(meta, compliance, metrics, primary, secondary_rows, sensitivity, baselines(y)),
        encoding="utf-8",
    )
    if make_figures:
        from plots import make_all
        make_all(out, y, metrics, primary, secondary_rows)
    return {"primary": primary, "metrics": metrics, "secondary": secondary_rows,
            "sensitivity": sensitivity, "meta": meta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(C.STAGES), required=True)
    args = parser.parse_args()
    data_dir = C.stage_dir(args.stage)
    if args.stage == "pilot":
        rows, bad = pilot_report(data_dir)
        print("PILOT_REPORT_OK", f"non-compliant: {len(bad)}", "->", data_dir / "results" / "pilot_summary.md")
        return
    analyse(data_dir)
    print("ANALYSIS_OK ->", data_dir / "results")


if __name__ == "__main__":
    main()
