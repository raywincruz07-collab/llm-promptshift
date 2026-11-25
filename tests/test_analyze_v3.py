"""analyze_v3: planted effects must be recovered; null effects must not be."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "promptshift"))

import analyze_v3 as V  # noqa: E402
import config as C  # noqa: E402
from common import write_jsonl  # noqa: E402

CONDS = C.CONDITIONS


def make(name, n, seed, flip_rate, accuracy, def_fixes, share=0.3, words=12, miss_rate=0.5):
    """Synthetic dataset. Predictions start from gold, then planted noise:
    - wording flips on W1/W2 for a share of posts (flip_rate),
    - errors with prob 1-accuracy (same for all conditions of a post),
    - without definition, offensive posts are missed extra often unless def_fixes is False."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < share).astype(int)
    base = np.where(rng.random(n) < accuracy, y, 1 - y)
    miss = (y == 1) & (rng.random(n) < miss_rate)     # under-labelling without the definition
    labels = {}
    for m in C.MODELS:
        cols = {}
        for c in CONDS:
            col = base.copy().astype(float)
            if c in ("W0", "W1", "W2", "R0"):
                col[miss] = 0
            if c in ("D0", "D1", "D2", "RD0") and not def_fixes:
                col[miss] = 0
            cols[c] = col
        flips = rng.random(n) < flip_rate
        for c in ("W1", "D1"):
            cols[c] = np.where(flips, 1 - cols[c], cols[c])
        labels[m] = cols
    sample = [{"row_id": f"{name}{i}", "text": " ".join(["w"] * (words + i % 7))} for i in range(n)]
    preds = []
    ds = V.Dataset(name, sample, y, labels, preds,
                   {"singleton": np.ones(n, bool), "annotators_agree": rng.random(n) > 0.1})
    return ds


def by_test(rows):
    return {r["test"]: r for r in rows}


def test_planted_effects_recovered():
    olid = make("OLID", 860, 1, flip_rate=0.02, accuracy=0.98, def_fixes=True)
    fresh = make("fresh", 400, 2, flip_rate=0.10, accuracy=0.90, def_fixes=True, miss_rate=0.8)
    r = by_test(V.primary(olid, fresh, n_boot=500))
    for m in ("Gemma", "Qwen", "Llama"):
        assert r[f"H1_{m}"]["estimate"] > 0.03 and r[f"H1_{m}"]["significant"]
        assert r[f"H2_{m}"]["estimate"] < -0.05 and r[f"H2_{m}"]["significant"]
        assert r[f"H3_{m}"]["estimate"] < -0.02 and r[f"H3_{m}"]["significant"]
    assert len(r) == 9 and all("p_holm" in v for v in r.values())


def test_null_effects_not_found():
    olid = make("OLID", 860, 3, flip_rate=0.05, accuracy=0.85, def_fixes=False)
    fresh = make("fresh", 400, 4, flip_rate=0.05, accuracy=0.85, def_fixes=False)
    r = by_test(V.primary(olid, fresh, n_boot=500))
    for m in ("Gemma", "Qwen", "Llama"):
        assert not r[f"H1_{m}"]["significant"]
        assert not r[f"H3_{m}"]["significant"]
        assert abs(r[f"H3_{m}"]["estimate"]) < 1e-9          # definition changes nothing here


def test_full_analysis_writes_outputs(tmp_path):
    olid = make("OLID", 300, 5, 0.03, 0.85, True)
    fresh = make("fresh", 150, 6, 0.06, 0.8, True)
    gold_meta = {"n_disagreements_adjudicated": 12,
                 "scores": {"fresh": {"krippendorff_alpha": 0.8, "raw_agreement": 0.92},
                            "olid_anchor": {"accuracy_vs_olid_A": 0.85, "accuracy_vs_olid_B": 0.8,
                                            "kappa_vs_olid_A": 0.65, "kappa_vs_olid_B": 0.55}}}
    V.analyse(olid, fresh, out=tmp_path, n_boot=200, gold_meta=gold_meta)
    md = (tmp_path / "summary_v3.md").read_text()
    assert "NOT met" in md                                    # kappa B 0.55 < 0.60 rule fires
    for name in ("primary_v3.csv", "metrics_by_condition_v3.csv", "secondary_v3.csv", "sensitivity_v3.csv",
                 "fig5_v3_primary_tests.png", "fig6_v3_ews_by_dataset.png", "fig7_v3_prevalence_by_dataset.png"):
        assert (tmp_path / name).exists(), name
    assert md.count("| H1_") >= 1 and "length" in md


def test_refuses_without_gold(tmp_path):
    write_jsonl(tmp_path / "sample.jsonl", [{"row_id": "a", "text": "x"}])
    with pytest.raises(SystemExit):
        V.load_fresh(tmp_path, tmp_path / "gold_main.jsonl")


def test_length_matching_is_a_subset_with_similar_lengths():
    olid = make("OLID", 860, 7, 0.03, 0.85, True, words=5)
    fresh = make("fresh", 400, 8, 0.03, 0.85, True, words=8)
    mask = V.length_matched_olid(olid, fresh)
    assert 0 < mask.sum() < olid.n
