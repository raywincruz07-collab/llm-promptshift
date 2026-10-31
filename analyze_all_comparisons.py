"""Recompute all nine planned held-out comparisons.

Bonferroni adjustment was added post hoc, after test results
were inspected. This is not a preregistered correction.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = DATA / "test_all9_comparisons.csv"

MODELS = ["gemma4-26b", "qwen3.6-36b", "llama-3.3-70b"]
PROMPTS = ["P0", "P1", "P2"]
LABELS = {"NOT_OFFENSIVE": 0, "OFFENSIVE": 1}

SEED = 6174
B = 5000
ALPHA = 0.05


def macro_f1(y, p):
    tn = np.sum((y == 0) & (p == 0))
    fp = np.sum((y == 0) & (p == 1))
    fn = np.sum((y == 1) & (p == 0))
    tp = np.sum((y == 1) & (p == 1))

    negative = 2 * tn / (2 * tn + fp + fn)
    positive = 2 * tp / (2 * tp + fp + fn)

    return (negative + positive) / 2


def effects(y, x):
    """Return six macro-F1 contrasts and three flip contrasts."""

    f1 = np.array([
        [macro_f1(y, x[:, i * 3 + j]) for j in range(3)]
        for i in range(3)
    ])

    flips = np.column_stack([
        np.ptp(x[:, i * 3:i * 3 + 3], axis=1) > 0
        for i in range(3)
    ]).mean(axis=0)

    values = []

    for i in range(3):
        for j in (1, 2):
            values.append(f1[i, j] - f1[i, 0])

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        values.append(flips[i] - flips[j])

    return np.array(values)


def main():
    assert not OUTPUT.exists(), "Output already exists"

    test = pd.read_parquet(DATA / "test860.parquet")
    pred = pd.read_json(
        DATA / "test_predictions.jsonl", lines=True
    )

    assert len(test) == 860
    assert len(pred) == 7740
    assert test["row_id"].is_unique
    assert not pred.duplicated(
        ["model", "row_id", "prompt_id"]
    ).any()

    columns = pd.MultiIndex.from_product(
        [MODELS, PROMPTS],
        names=["model", "prompt_id"],
    )

    wide = pred.pivot(
        index="row_id",
        columns=["model", "prompt_id"],
        values="raw_response",
    ).reindex(
        index=test["row_id"],
        columns=columns,
    )

    numeric = wide.apply(lambda s: s.map(LABELS))
    assert numeric.shape == (860, 9)
    assert numeric.notna().all().all()

    x = numeric.to_numpy(dtype=int)
    y = test.set_index("row_id").loc[
        wide.index, "label"
    ].to_numpy(dtype=int)

    assert np.sum(y == 0) == 620
    assert np.sum(y == 1) == 240

    observed = effects(y, x)

    rng = np.random.default_rng(SEED)
    negative_ids = np.flatnonzero(y == 0)
    positive_ids = np.flatnonzero(y == 1)

    boot = np.empty((B, 9))

    for b in range(B):
        selected = np.concatenate([
            rng.choice(
                negative_ids, len(negative_ids), replace=True
            ),
            rng.choice(
                positive_ids, len(positive_ids), replace=True
            ),
        ])

        boot[b] = effects(y[selected], x[selected])

    names = [
        f"{model}: {prompt} minus P0"
        for model in MODELS
        for prompt in ("P1", "P2")
    ]

    names += [
        f"{MODELS[i]} minus {MODELS[j]}: flip rate"
        for i, j in [(0, 1), (0, 2), (1, 2)]
    ]

    # Nine-comparison Bonferroni adjustment.
    # This correction was chosen after inspecting test results.
    adjusted_tail = ALPHA / (2 * 9)
    rows = []

    for k, name in enumerate(names):
        ordinary = np.quantile(
            boot[:, k], [0.025, 0.975]
        )

        adjusted = np.quantile(
            boot[:, k],
            [adjusted_tail, 1 - adjusted_tail],
        )

        rows.append({
            "comparison": name,
            "effect_type": (
                "macro_f1_difference"
                if k < 6 else "flip_rate_difference"
            ),
            "observed_difference": observed[k],
            "unadjusted_95_low": ordinary[0],
            "unadjusted_95_high": ordinary[1],
            "bonferroni_9_low": adjusted[0],
            "bonferroni_9_high": adjusted[1],
            "adjusted_ci_excludes_zero": bool(
                adjusted[0] > 0 or adjusted[1] < 0
            ),
            "adjustment_status": "post_hoc",
        })

    result = pd.DataFrame(rows)

    assert len(result) == 9
    assert np.isclose(
        result.iloc[5]["observed_difference"],
        0.00909,
        atol=0.0001,
    )

    result.to_csv(OUTPUT, index=False)

    print("ALL_9_COMPARISONS_OK")
    print("Comparisons:", len(result))
    print("Bootstrap resamples:", B)
    print("Correction: Bonferroni across 9 (POST HOC)")
    print()
    print(result.round(5).to_string(index=False))
    print()
    print(
        "Adjusted intervals excluding zero:",
        int(result["adjusted_ci_excludes_zero"].sum()),
    )
    print("Output:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
