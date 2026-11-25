"""PromptShift v3 analysis: OLID (likely seen) vs fresh Bluesky posts (unseen). PLAN_v3.md, section 6.

    python src/promptshift/analyze_v3.py

Refuses to run until (a) every OLID and fresh request has a result and (b) the fresh gold labels
exist (annotate_fresh.py finalize). Writes numbers-only tables and figures to data/fresh/results_v3/.

Primary family (9 tests, Holm, alpha 0.05), per model:
  H1  Delta EWS = EWS_fresh - EWS_OLID > 0,   EWS = mean_post[ pairwise disagreement W0..W2 - 1(W0 != R0) ]
  H2  Delta F1  = F1_fresh  - F1_OLID  < 0,   F1  = mean over W0..W2, D0..D2 of macro-F1
  H3  Delta PE  = PE_D - PE_W < 0 on fresh,   PE_X = mean over wordings |pred share - gold share|
H3 on OLID is descriptive only (already observed before the plan was frozen).

Code drafted with AI assistance (Claude, Anthropic); the team reviewed it and
is responsible for it (course AI-use policy).
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import stats as S  # noqa: E402
from analyze import complete_rows, singleton_mask  # noqa: E402
from common import git_state, read_json, read_jsonl, sha256_file, utc_now, write_json  # noqa: E402

W = ["W0", "W1", "W2"]
D = ["D0", "D1", "D2"]
SESOI_V3 = {"H1": 0.01, "H2": 0.03, "H3": 0.02}
KAPPA_FLOOR = 0.60
SEED = 20261201
OUT = C.DATA_DIR / "fresh" / "results_v3"


# ------------------------------------------------------------------ loading
class Dataset:
    """Gold labels y (n,), predictions labels[model][condition] (n,) with NaN for invalid."""

    def __init__(self, name, sample, y, labels, predictions, extra=None):
        self.name, self.sample, self.y, self.labels = name, sample, np.asarray(y, int), labels
        self.predictions, self.extra = predictions, extra or {}

    @property
    def n(self):
        return len(self.y)


def _labels_from_predictions(sample, predictions, requests, allow_incomplete=False):
    missing = len(requests) - len(predictions)
    if missing and not allow_incomplete:
        raise SystemExit(f"{missing} requests still have no result. Finish the run first.")
    position = {r["row_id"]: i for i, r in enumerate(sample)}
    labels = {m: {c: np.full(len(sample), np.nan) for c in C.CONDITIONS} for m in C.MODELS}
    for p in predictions:
        value = C.LABELS.get(p.get("parsed_label")) if p["compliant"] else None
        if value is not None and p["row_id"] in position:
            labels[p["model"]][p["condition"]][position[p["row_id"]]] = value
    return labels


def load_olid(data_dir=None):
    data_dir = Path(data_dir or C.stage_dir("main"))
    sample = read_jsonl(data_dir / "sample.jsonl")
    predictions = read_jsonl(data_dir / "predictions.jsonl")
    labels = _labels_from_predictions(sample, predictions, read_jsonl(data_dir / "requests.jsonl"))
    return Dataset("OLID", sample, [r["label"] for r in sample], labels, predictions,
                   {"singleton": singleton_mask(sample)})


def load_fresh(data_dir=None, gold_path=None):
    data_dir = Path(data_dir or C.stage_dir("fresh"))
    gold_path = Path(gold_path or C.DATA_DIR / "fresh" / "gold_main.jsonl")
    gold = {r["row_id"]: r for r in read_jsonl(gold_path)}
    if not gold:
        raise SystemExit(f"{gold_path} not found: run annotate_fresh.py finalize first (blinding rule).")
    sample = read_jsonl(data_dir / "sample.jsonl")
    missing = [r["row_id"] for r in sample if r["row_id"] not in gold]
    if missing:
        raise SystemExit(f"{len(missing)} fresh posts have no gold label, e.g. {missing[:5]}")
    predictions = read_jsonl(data_dir / "predictions.jsonl")
    labels = _labels_from_predictions(sample, predictions, read_jsonl(data_dir / "requests.jsonl"))
    y = [gold[r["row_id"]]["label"] for r in sample]
    agree = np.array([bool(gold[r["row_id"]].get("annotators_agree", True)) for r in sample])
    return Dataset("fresh", sample, y, labels, predictions, {"annotators_agree": agree})


# ------------------------------------------------------------------ estimands
def ews_per_post(ds, model, invalid_as_disagreement=False, subset=None):
    cols = {c: ds.labels[model][c] for c in W + ["R0"]}
    if invalid_as_disagreement:
        keep = np.ones(ds.n, bool) if subset is None else subset
        cols = {c: np.where(np.isnan(v), -1 - i, v)[keep] for i, (c, v) in enumerate(cols.items())}
    else:
        keep = complete_rows(ds.labels, model, W + ["R0"])
        if subset is not None:
            keep &= subset
        cols = {c: v[keep] for c, v in cols.items()}
    return S.mean_pairwise_disagreement([cols[c] for c in W]) - (cols["W0"] != cols["R0"]).astype(float)


def _cols(ds, model, conditions, invalid_as_error=False, subset=None):
    if invalid_as_error:
        keep = np.ones(ds.n, bool) if subset is None else subset
        cols = {c: np.where(np.isnan(ds.labels[model][c]), 1 - ds.y, ds.labels[model][c])[keep].astype(int)
                for c in conditions}
    else:
        keep = complete_rows(ds.labels, model, conditions)
        if subset is not None:
            keep &= subset
        cols = {c: ds.labels[model][c][keep].astype(int) for c in conditions}
    return ds.y[keep], cols


def f1_avg(y, cols, conditions, idx=None):
    if idx is None:
        return float(np.mean([S.macro_f1(y, cols[c]) for c in conditions]))
    yb = y[idx]
    return np.mean([S.macro_f1(yb, cols[c][idx]) for c in conditions], axis=0)


def prevalence_error(y, cols, conditions, idx=None):
    if idx is None:
        return float(np.mean([abs(S.prevalence(cols[c]) - S.prevalence(y)) for c in conditions]))
    yb = y[idx]
    return np.mean([np.abs(S.prevalence(cols[c][idx]) - S.prevalence(yb)) for c in conditions], axis=0)


def plain_indices(n, n_boot, rng):
    return rng.integers(0, n, size=(n_boot, n)).astype(np.int32)


# ------------------------------------------------------------------ tests
def test_h1(olid, fresh, model, rng, n_boot, **opt):
    a = ews_per_post(fresh, model, opt.get("invalid", False), opt.get("fresh_subset"))
    b = ews_per_post(olid, model, opt.get("invalid", False), opt.get("olid_subset"))
    boot = (a[plain_indices(len(a), n_boot, rng)].mean(axis=1)
            - b[plain_indices(len(b), n_boot, rng)].mean(axis=1))
    r = S.summarise(a.mean() - b.mean(), boot, alternative="greater")
    return r | {"n_fresh": len(a), "n_olid": len(b), "fresh_value": float(a.mean()), "olid_value": float(b.mean())}


def test_h2(olid, fresh, model, rng, n_boot, **opt):
    conds = W + D
    yf, cf = _cols(fresh, model, conds, opt.get("invalid", False), opt.get("fresh_subset"))
    yo, co = _cols(olid, model, conds, opt.get("invalid", False), opt.get("olid_subset"))
    obs_f, obs_o = f1_avg(yf, cf, conds), f1_avg(yo, co, conds)
    boot = (f1_avg(yf, cf, conds, S.stratified_indices(yf, n_boot, rng))
            - f1_avg(yo, co, conds, S.stratified_indices(yo, n_boot, rng)))
    r = S.summarise(obs_f - obs_o, boot, alternative="less")
    return r | {"n_fresh": len(yf), "n_olid": len(yo), "fresh_value": obs_f, "olid_value": obs_o}


def test_h3(ds, model, rng, n_boot, **opt):
    y, cols = _cols(ds, model, W + D, opt.get("invalid", False), opt.get("subset"))
    idx = S.stratified_indices(y, n_boot, rng)
    pe_d, pe_w = prevalence_error(y, cols, D), prevalence_error(y, cols, W)
    boot = prevalence_error(y, cols, D, idx) - prevalence_error(y, cols, W, idx)
    r = S.summarise(pe_d - pe_w, boot, alternative="less")
    return r | {"n": len(y), "pe_definition": pe_d, "pe_no_definition": pe_w,
                "gold_share": float(S.prevalence(y))}


def verdict(r, sesoi):
    if r["alternative"] == "greater":
        return S.size_verdict(r, sesoi)
    if r["one_sided_upper"] < -sesoi:
        return "materially negative (upper bound below -SESOI)"
    if r["one_sided_lower"] > -sesoi:
        return "smaller than SESOI (lower bound above -SESOI)"
    return "size inconclusive"


def primary(olid, fresh, n_boot, seed=SEED, **opt):
    rows = []
    for k, model in enumerate(C.MODELS):
        short = C.MODEL_SHORT[model]
        r1 = test_h1(olid, fresh, model, np.random.default_rng([seed, 1, k]), n_boot, **opt)
        rows.append({"test": f"H1_{short}", "model": model, "sesoi": SESOI_V3["H1"],
                     "estimand": "EWS fresh - EWS OLID", **r1})
        r2 = test_h2(olid, fresh, model, np.random.default_rng([seed, 2, k]), n_boot, **opt)
        rows.append({"test": f"H2_{short}", "model": model, "sesoi": SESOI_V3["H2"],
                     "estimand": "mean macro-F1 fresh - OLID", **r2})
        r3 = test_h3(fresh, model, np.random.default_rng([seed, 3, k]), n_boot,
                     invalid=opt.get("invalid", False), subset=opt.get("fresh_subset"))
        rows.append({"test": f"H3_{short}", "model": model, "sesoi": SESOI_V3["H3"],
                     "estimand": "prevalence error D - W (fresh)", **r3})
    holm = S.holm([r["p_value"] for r in rows])
    for r, p in zip(rows, holm):
        r["p_holm"] = float(p)
        r["significant"] = bool(p < 0.05)
        r["size_verdict"] = verdict(r, r["sesoi"])
    return rows


# ------------------------------------------------------------------ secondary
def condition_table(ds, rng, n_boot):
    rows = []
    for model in C.MODELS:
        for c in C.CONDITIONS:
            col = ds.labels[model][c]
            valid = ~np.isnan(col)
            y, p = ds.y[valid], col[valid].astype(int)
            idx = S.stratified_indices(y, n_boot, rng)
            f1b = S.macro_f1(y[idx], p[idx])
            rows.append({"dataset": ds.name, "model": model, "condition": c, "n_valid": int(valid.sum()),
                         "accuracy": float(S.accuracy(y, p)), "macro_f1": float(S.macro_f1(y, p)),
                         "macro_f1_ci_low": float(np.quantile(f1b, 0.025)),
                         "macro_f1_ci_high": float(np.quantile(f1b, 0.975)),
                         "offensive_recall": float(S.recall_offensive(y, p)),
                         "not_offensive_recall": float(S.recall_not_offensive(y, p)),
                         "predicted_share": float(S.prevalence(p)), "gold_share": float(S.prevalence(y))})
    return rows


def _mean_ci(values, rng, n_boot):
    values = np.asarray(values, float)
    boot = values[plain_indices(len(values), n_boot, rng)].mean(axis=1)
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def any_flip(ds, model):
    keep = complete_rows(ds.labels, model, W)
    cols = [ds.labels[model][c] for c in W]
    flip = np.full(ds.n, np.nan)
    flip[keep] = (S.mean_pairwise_disagreement([c[keep] for c in cols]) > 0).astype(float)
    return flip


def first_token_confidence(ds, model, condition="W0"):
    conf = np.full(ds.n, np.nan)
    position = {r["row_id"]: i for i, r in enumerate(ds.sample)}
    for p in ds.predictions:
        if p["model"] == model and p["condition"] == condition and p.get("logprobs"):
            lp = p["logprobs"][0].get("logprob")
            if lp is not None and math.isfinite(lp):
                conf[position[p["row_id"]]] = math.exp(lp)
    return conf


def secondary(olid, fresh, n_boot, seed=SEED):
    rows = []

    def add(dataset, model, measure, est, lo=None, hi=None, n=None, note=""):
        rows.append({"dataset": dataset, "model": model, "measure": measure, "estimate": est,
                     "ci_low": lo, "ci_high": hi, "n": n, "note": note})

    for k, model in enumerate(C.MODELS):
        for j, ds in enumerate((olid, fresh)):
            rng = np.random.default_rng([seed, 10, k, j])
            keep = complete_rows(ds.labels, model, W + ["R0"])
            wd = S.mean_pairwise_disagreement([ds.labels[model][c][keep] for c in W])
            rd = (ds.labels[model]["W0"][keep] != ds.labels[model]["R0"][keep]).astype(float)
            keep_d = complete_rows(ds.labels, model, D + ["RD0"])
            rdd = (ds.labels[model]["D0"][keep_d] != ds.labels[model]["RD0"][keep_d]).astype(float)
            add(ds.name, model, "wording disagreement (mean pairwise W0-W2)", *_mean_ci(wd, rng, n_boot), len(wd))
            add(ds.name, model, "repeat disagreement W0 vs R0", *_mean_ci(rd, rng, n_boot), len(rd))
            add(ds.name, model, "repeat disagreement D0 vs RD0", *_mean_ci(rdd, rng, n_boot), len(rdd))
            add(ds.name, model, "any-flip rate across W0-W2", *_mean_ci(wd > 0, rng, n_boot), len(wd))
            if ds is olid:
                r = test_h3(olid, model, rng, n_boot)
                add("OLID", model, "H3 on OLID (descriptive, not confirmatory): PE_D - PE_W",
                    r["estimate"], r["ci_low"], r["ci_high"], r["n"], "observed before freeze")
            # E2: model confidence (first-token probability, W0) for flipped vs stable posts
            flip, conf = any_flip(ds, model), first_token_confidence(ds, model)
            ok = ~np.isnan(flip) & ~np.isnan(conf)
            if ok.sum() and flip[ok].sum() and (1 - flip[ok]).sum():
                a, b = conf[ok][flip[ok] == 1], conf[ok][flip[ok] == 0]
                d = float(a.mean() - b.mean())
                ba = a[plain_indices(len(a), n_boot, rng)].mean(axis=1)
                bb = b[plain_indices(len(b), n_boot, rng)].mean(axis=1)
                add(ds.name, model, "E2 confidence (W0 first-token prob): flipped - stable", d,
                    float(np.quantile(ba - bb, 0.025)), float(np.quantile(ba - bb, 0.975)), int(ok.sum()), "exploratory")
            elif not ok.sum():
                add(ds.name, model, "E2 confidence", None, note="no log probabilities recorded")
        # E1: difference-in-differences of the definition effect on macro-F1
        rng = np.random.default_rng([seed, 11, k])
        conds = W + D
        yf, cf = _cols(fresh, model, conds)
        yo, co = _cols(olid, model, conds)
        did = (f1_avg(yf, cf, D) - f1_avg(yf, cf, W)) - (f1_avg(yo, co, D) - f1_avg(yo, co, W))
        i_f, i_o = S.stratified_indices(yf, n_boot, rng), S.stratified_indices(yo, n_boot, rng)
        boot = ((f1_avg(yf, cf, D, i_f) - f1_avg(yf, cf, W, i_f))
                - (f1_avg(yo, co, D, i_o) - f1_avg(yo, co, W, i_o)))
        add("fresh - OLID", model, "E1 definition effect on macro-F1, fresh minus OLID", float(did),
            float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), len(yf) + len(yo), "exploratory")
        # E4: flip rate where annotators disagreed vs agreed (fresh)
        flip = any_flip(fresh, model)
        agree = fresh.extra["annotators_agree"]
        ok = ~np.isnan(flip)
        a, b = flip[ok & ~agree], flip[ok & agree]
        if len(a) and len(b):
            ba = a[plain_indices(len(a), n_boot, rng)].mean(axis=1)
            bb = b[plain_indices(len(b), n_boot, rng)].mean(axis=1)
            add("fresh", model, "E4 any-flip rate: annotators disagreed - agreed", float(a.mean() - b.mean()),
                float(np.quantile(ba - bb, 0.025)), float(np.quantile(ba - bb, 0.975)), int(ok.sum()),
                f"exploratory; {len(a)} posts with annotator disagreement")
    # E3: between-model disagreement on W0
    for j, ds in enumerate((olid, fresh)):
        for a_i in range(len(C.MODELS)):
            for b_i in range(a_i + 1, len(C.MODELS)):
                ma, mb = C.MODELS[a_i], C.MODELS[b_i]
                x, z = ds.labels[ma]["W0"], ds.labels[mb]["W0"]
                ok = ~np.isnan(x) & ~np.isnan(z)
                rng = np.random.default_rng([seed, 12, j, a_i, b_i])
                add(ds.name, f"{C.MODEL_SHORT[ma]} vs {C.MODEL_SHORT[mb]}", "E3 between-model disagreement on W0",
                    *_mean_ci(x[ok] != z[ok], rng, n_boot), int(ok.sum()), "exploratory")
    return rows


def length_matched_olid(olid, fresh, seed=SEED):
    """OLID subset whose word-count distribution matches the fresh posts (fresh quintile bins)."""
    lf = np.array([len(r["text"].split()) for r in fresh.sample])
    lo = np.array([len(r["text"].split()) for r in olid.sample])
    edges = np.unique(np.quantile(lf, [0.2, 0.4, 0.6, 0.8]))
    n_bins = len(edges) + 1
    bf, bo = np.digitize(lf, edges), np.digitize(lo, edges)
    share = np.bincount(bf, minlength=n_bins) / len(lf)
    avail = np.bincount(bo, minlength=n_bins)
    usable = (share > 0) & (avail > 0)          # fresh length bins with no OLID post cannot be matched
    if not usable.any():
        return np.ones(len(lo), bool)
    share = np.where(usable, share, 0) / share[usable].sum()
    total = min(len(lo), *[avail[b] / share[b] for b in range(n_bins) if usable[b]])
    rng = np.random.default_rng([seed, 20])
    mask = np.zeros(len(lo), bool)
    for b in range(n_bins):
        k = int(np.floor(share[b] * total))
        members = np.flatnonzero(bo == b)
        if k and len(members):
            mask[rng.choice(members, size=min(k, len(members)), replace=False)] = True
    return mask


def sensitivity(olid, fresh, n_boot):
    analyses = {
        "invalid answers counted as disagreement/error": {"invalid": True},
        "OLID exact duplicates removed": {"olid_subset": olid.extra["singleton"]},
        "fresh posts with annotator disagreement removed": {"fresh_subset": fresh.extra["annotators_agree"]},
        "OLID subsample matched on post length": {"olid_subset": length_matched_olid(olid, fresh)},
    }
    rows = []
    for name, opt in analyses.items():
        for r in primary(olid, fresh, n_boot, **opt):
            rows.append({"analysis": name, **{k: r[k] for k in ("test", "estimate", "ci_low", "ci_high",
                                                                 "p_holm", "size_verdict")},
                         "n_fresh": r.get("n_fresh", r.get("n")), "n_olid": r.get("n_olid")})
    return rows


# ------------------------------------------------------------------ figures
def figures(out, prim, olid, fresh):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plots import INK_2, MODEL_COLOR, MODEL_MARKER, _save, _style

    # Forest plot of the 9 primary tests, one panel per hypothesis
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    titles = {"H1": "H1: excess wording sensitivity\nfresh − OLID (higher = more sensitive on fresh)",
              "H2": "H2: mean macro-F1\nfresh − OLID (negative = worse on fresh)",
              "H3": "H3: prevalence error, fresh\ndefinition − no definition (negative = definition helps)"}
    for ax, h in zip(axes, ("H1", "H2", "H3")):
        rs = [r for r in prim if r["test"].startswith(h)]
        for i, r in enumerate(rs):
            m = r["model"]
            ax.errorbar(r["estimate"], i, xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                        fmt=MODEL_MARKER[m], color=MODEL_COLOR[m], elinewidth=2, markersize=7)
            ax.annotate(f"{C.MODEL_SHORT[m]}  {r['estimate']:+.3f}{' *' if r['significant'] else ''}",
                        (1.0, i), xycoords=("axes fraction", "data"), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK_2)
        s = SESOI_V3[h] * (1 if h == "H1" else -1)
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.axvline(s, color=INK_2, lw=0.8, ls="--")
        ax.set_yticks([])
        ax.set_ylim(-0.7, len(rs) - 0.3)
        ax.set_title(titles[h], fontsize=9.5)
        _style(ax)
    fig.text(0.01, -0.02, "Bars: 95% bootstrap CI. Dashed: smallest effect of interest. * = significant after Holm (9 tests).",
             fontsize=8, color=INK_2, ha="left")
    _save(fig, out, "fig5_v3_primary_tests", w_pad=6)

    # Wording vs repeat disagreement by dataset
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for k, model in enumerate(C.MODELS):
        for j, ds in enumerate((olid, fresh)):
            v = ews_per_post(ds, model)
            ax.errorbar(k * 3 + j, v.mean(), yerr=1.96 * v.std(ddof=1) / math.sqrt(len(v)),
                        fmt=MODEL_MARKER[model], color=MODEL_COLOR[model], elinewidth=2, markersize=7,
                        markerfacecolor="white" if ds is olid else MODEL_COLOR[model])
    ax.set_xticks([k * 3 + j for k in range(3) for j in range(2)])
    ax.set_xticklabels([f"{C.MODEL_SHORT[m]}\n{d}" for m in C.MODELS for d in ("OLID", "fresh")], fontsize=8)
    ax.set_ylabel("excess wording sensitivity")
    ax.set_title("Wording minus repeat disagreement per post (hollow = OLID, filled = fresh; bars: 95% CI)",
                 fontsize=9.5)
    ax.axhline(0, color=INK_2, lw=0.8)
    _style(ax, "y")
    _save(fig, out, "fig6_v3_ews_by_dataset")

    # Predicted vs gold offensive share, by dataset and condition set
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, ds in zip(axes, (olid, fresh)):
        gold = float(np.mean(ds.y))
        for k, model in enumerate(C.MODELS):
            for j, conds in enumerate((W, D)):
                share = np.nanmean([np.nanmean(ds.labels[model][c]) for c in conds])
                ax.plot(k + (j - 0.5) * 0.25, share, MODEL_MARKER[model], color=MODEL_COLOR[model], markersize=8,
                        markerfacecolor="white" if j == 0 else MODEL_COLOR[model])
        ax.axhline(gold, color=INK_2, lw=1, ls="--")
        ax.annotate(f"human share {gold:.2f}", (0.5, gold), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
        ax.set_xticks(range(3))
        ax.set_xticklabels([C.MODEL_SHORT[m] for m in C.MODELS])
        ax.set_title(f"{ds.name}: share labelled offensive\n(hollow = no definition, filled = definition)", fontsize=9.5)
        _style(ax, "y")
    _save(fig, out, "fig7_v3_prevalence_by_dataset")


# ------------------------------------------------------------------ output
def _write_csv(path, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with Path(path).open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _f(x, fmt="+.4f"):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def summary_md(prim, cond_rows, sec, sens, olid, fresh, gold_meta):
    scores = (gold_meta or {}).get("scores", {}).get("olid_anchor", {})
    kappas = [scores.get("kappa_vs_olid_A"), scores.get("kappa_vs_olid_B")]
    h2_ok = all(k is not None and k >= KAPPA_FLOOR for k in kappas)
    lines = ["# PromptShift v3 results (generated tables, no interpretation)", "",
             f"Generated {utc_now()}. OLID n={olid.n} (gold share {np.mean(olid.y):.3f}); "
             f"fresh n={fresh.n} (gold share {np.mean(fresh.y):.3f}). Bootstrap B={C.N_BOOTSTRAP}, seed {SEED}. "
             "Holm across 9 primary tests, alpha 0.05.", "",
             "## Gold labels (fresh)", ""]
    if gold_meta:
        fs = gold_meta["scores"]["fresh"]
        lines += [f"Krippendorff alpha (A vs B, fresh): {fs['krippendorff_alpha']:.3f}; raw agreement "
                  f"{fs['raw_agreement']:.3f}; {gold_meta['n_disagreements_adjudicated']} disagreements decided by C.",
                  f"Agreement with OLID gold on 100 anchors: A accuracy {scores.get('accuracy_vs_olid_A', float('nan')):.3f} "
                  f"(kappa {kappas[0]:.3f}), B accuracy {scores.get('accuracy_vs_olid_B', float('nan')):.3f} "
                  f"(kappa {kappas[1]:.3f}). Rule fixed in advance: H2 interpretable only if both kappas ≥ {KAPPA_FLOOR}: "
                  f"**{'met' if h2_ok else 'NOT met — H2 is not interpretable as a contamination effect'}**.", ""]
    lines += ["## Primary tests", "",
              "| Test | Estimate | 95% CI | p | p (Holm) | Significant | Size verdict | fresh / OLID value | n fresh / OLID |",
              "|---|---:|---|---:|---:|---|---|---|---|"]
    for r in prim:
        vals = (f"{_f(r.get('fresh_value'), '.4f')} / {_f(r.get('olid_value'), '.4f')}" if "fresh_value" in r
                else f"PE_D {r['pe_definition']:.4f} / PE_W {r['pe_no_definition']:.4f}")
        ns = f"{r.get('n_fresh', r.get('n'))} / {r.get('n_olid', '')}"
        lines.append(f"| {r['test']} | {r['estimate']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                     f"{r['p_value']:.4f} | {r['p_holm']:.4f} | {r['significant']} | {r['size_verdict']} | {vals} | {ns} |")
    lines += ["", "## Metrics by dataset and condition", "",
              "| Dataset | Model | Condition | n | Accuracy | Macro-F1 [95% CI] | Off. recall | Not-off. recall | Predicted share | Gold share |",
              "|---|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for r in cond_rows:
        lines.append(f"| {r['dataset']} | {C.MODEL_SHORT[r['model']]} | {r['condition']} | {r['n_valid']} | "
                     f"{r['accuracy']:.4f} | {r['macro_f1']:.4f} [{r['macro_f1_ci_low']:.4f}, {r['macro_f1_ci_high']:.4f}] | "
                     f"{r['offensive_recall']:.4f} | {r['not_offensive_recall']:.4f} | {r['predicted_share']:.4f} | {r['gold_share']:.4f} |")
    lines += ["", "## Secondary and exploratory measures", "",
              "| Dataset | Model | Measure | Estimate | 95% CI | n | Note |", "|---|---|---|---:|---|---:|---|"]
    for r in sec:
        model = C.MODEL_SHORT.get(r["model"], r["model"])
        ci = f"[{_f(r['ci_low'])}, {_f(r['ci_high'])}]" if r["ci_low"] is not None else ""
        lines.append(f"| {r['dataset']} | {model} | {r['measure']} | {_f(r['estimate'])} | {ci} | {r['n'] or ''} | {r['note']} |")
    lines += ["", "## Sensitivity analyses (primary estimands; Holm within each analysis)", "",
              "| Analysis | Test | Estimate | 95% CI | p (Holm) | Size verdict | n fresh / OLID |", "|---|---|---:|---|---:|---|---|"]
    for r in sens:
        lines.append(f"| {r['analysis']} | {r['test']} | {r['estimate']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                     f"{r['p_holm']:.4f} | {r['size_verdict']} | {r['n_fresh']} / {r['n_olid'] or ''} |")
    return "\n".join(lines) + "\n"


def analyse(olid, fresh, out=OUT, n_boot=None, gold_meta=None, make_figures=True):
    n_boot = n_boot or C.N_BOOTSTRAP
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prim = primary(olid, fresh, n_boot)
    rng = np.random.default_rng([SEED, 30])
    cond_rows = condition_table(olid, rng, n_boot) + condition_table(fresh, rng, n_boot)
    sec = secondary(olid, fresh, n_boot)
    sens = sensitivity(olid, fresh, n_boot)
    _write_csv(out / "primary_v3.csv", prim)
    _write_csv(out / "metrics_by_condition_v3.csv", cond_rows)
    _write_csv(out / "secondary_v3.csv", sec)
    _write_csv(out / "sensitivity_v3.csv", sens)
    (out / "summary_v3.md").write_text(summary_md(prim, cond_rows, sec, sens, olid, fresh, gold_meta), encoding="utf-8")
    if make_figures:
        figures(out, prim, olid, fresh)
    return prim, cond_rows, sec, sens


def main():
    olid, fresh = load_olid(), load_fresh()
    meta_path = C.DATA_DIR / "fresh" / "gold_meta.json"
    gold_meta = read_json(meta_path) if meta_path.exists() else None
    analyse(olid, fresh, gold_meta=gold_meta)
    write_json(OUT / "run_meta.json", {
        "created_utc": utc_now(), "git": git_state(C.ROOT), "seed": SEED, "n_bootstrap": C.N_BOOTSTRAP,
        "olid_predictions_sha256": sha256_file(C.stage_dir("main") / "predictions.jsonl"),
        "fresh_predictions_sha256": sha256_file(C.stage_dir("fresh") / "predictions.jsonl"),
        "gold_sha256": sha256_file(C.DATA_DIR / "fresh" / "gold_main.jsonl")})
    print("ANALYSIS_V3_OK ->", OUT)


if __name__ == "__main__":
    main()
