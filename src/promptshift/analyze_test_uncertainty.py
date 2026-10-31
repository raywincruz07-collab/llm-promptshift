"""Bootstrap uncertainty analysis for the IS6170 main experiment."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

MODELS = ["gemma4-26b", "qwen3.6-36b", "llama-3.3-70b"]
PROMPTS = ["P0", "P1", "P2"]

N_BOOTSTRAP = 5000
SEED = 6174

LABEL_MAP = {
    "NOT_OFFENSIVE": 0,
    "OFFENSIVE": 1,
}


def macro_f1(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    f1_negative = 2 * tn / (2 * tn + fp + fn)
    f1_positive = 2 * tp / (2 * tp + fp + fn)

    return (f1_negative + f1_positive) / 2


def confidence_interval(values):
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def main():
    posts = pd.read_parquet(DATA / "test860.parquet")
    predictions = pd.read_json(
        DATA / "test_predictions.jsonl",
        lines=True,
    )

    assert len(posts) == 860
    assert len(predictions) == 7740

    predictions["predicted"] = predictions[
        "raw_response"
    ].map(LABEL_MAP)

    assert predictions["predicted"].notna().all()

    columns = pd.MultiIndex.from_product(
        [MODELS, PROMPTS],
        names=["model", "prompt_id"],
    )

    wide = predictions.pivot(
        index="row_id",
        columns=["model", "prompt_id"],
        values="predicted",
    )

    wide = wide.reindex(
        index=posts["row_id"],
        columns=columns,
    )

    assert wide.shape == (860, 9)
    assert not wide.isna().any().any()

    y = posts.set_index("row_id").loc[
        wide.index, "label"
    ].to_numpy(dtype=int)

    x = wide.to_numpy(dtype=int)

    assert np.sum(y == 0) == 620
    assert np.sum(y == 1) == 240

    # Each row represents one post.
    # Each column represents one model-prompt combination.
    flips = np.column_stack([
        np.ptp(x[:, i * 3:i * 3 + 3], axis=1)
        for i in range(3)
    ])

    observed_flip = flips.mean(axis=0)

    observed_f1 = np.array([
        [
            macro_f1(y, x[:, i * 3 + j])
            for j in range(3)
        ]
        for i in range(3)
    ])

    rng = np.random.default_rng(SEED)

    negative_ids = np.flatnonzero(y == 0)
    positive_ids = np.flatnonzero(y == 1)

    bootstrap_flips = np.empty((N_BOOTSTRAP, 3))
    bootstrap_f1 = np.empty((N_BOOTSTRAP, 3, 3))

    for b in range(N_BOOTSTRAP):
        # Resample posts within each gold-label class.
        selected = np.concatenate([
            rng.choice(
                negative_ids,
                size=len(negative_ids),
                replace=True,
            ),
            rng.choice(
                positive_ids,
                size=len(positive_ids),
                replace=True,
            ),
        ])

        bootstrap_flips[b] = flips[selected].mean(axis=0)

        sampled_gold = y[selected]

        for i in range(3):
            for j in range(3):
                sampled_predictions = x[
                    selected, i * 3 + j
                ]

                bootstrap_f1[b, i, j] = macro_f1(
                    sampled_gold,
                    sampled_predictions,
                )

    # Confidence intervals for prompt-flip rates.
    flip_rows = []

    for i, model in enumerate(MODELS):
        low, high = confidence_interval(
            bootstrap_flips[:, i]
        )

        flip_rows.append({
            "model": model,
            "flip_rate": observed_flip[i],
            "ci_low": low,
            "ci_high": high,
        })

    # Confidence intervals for each macro-F1 score.
    f1_rows = []

    for i, model in enumerate(MODELS):
        for j, prompt_id in enumerate(PROMPTS):
            low, high = confidence_interval(
                bootstrap_f1[:, i, j]
            )

            f1_rows.append({
                "model": model,
                "prompt_id": prompt_id,
                "macro_f1": observed_f1[i, j],
                "ci_low": low,
                "ci_high": high,
            })

    # Paired macro-F1 differences relative to P0.
    delta_rows = []

    for i, model in enumerate(MODELS):
        for j in (1, 2):
            differences = (
                bootstrap_f1[:, i, j]
                - bootstrap_f1[:, i, 0]
            )

            low, high = confidence_interval(differences)

            delta_rows.append({
                "model": model,
                "comparison": f"{PROMPTS[j]} minus P0",
                "macro_f1_difference": (
                    observed_f1[i, j] - observed_f1[i, 0]
                ),
                "ci_low": low,
                "ci_high": high,
            })

    # Paired differences in flip rates between models.
    model_delta_rows = []

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        differences = (
            bootstrap_flips[:, i] - bootstrap_flips[:, j]
        )

        low, high = confidence_interval(differences)

        model_delta_rows.append({
            "comparison": f"{MODELS[i]} minus {MODELS[j]}",
            "flip_rate_difference": (
                observed_flip[i] - observed_flip[j]
            ),
            "ci_low": low,
            "ci_high": high,
        })

    flip_df = pd.DataFrame(flip_rows)
    f1_df = pd.DataFrame(f1_rows)
    delta_df = pd.DataFrame(delta_rows)
    model_delta_df = pd.DataFrame(model_delta_rows)

    flip_df.to_csv(
        DATA / "test_flip_confidence.csv", index=False
    )
    f1_df.to_csv(
        DATA / "test_f1_confidence.csv", index=False
    )
    delta_df.to_csv(
        DATA / "test_f1_differences.csv", index=False
    )
    model_delta_df.to_csv(
        DATA / "test_model_flip_differences.csv", index=False
    )

    print("TEST_UNCERTAINTY_OK")

    print("\nPROMPT-FLIP RATE CONFIDENCE INTERVALS:")
    print(flip_df.round(4).to_string(index=False))

    print("\nMACRO-F1 CONFIDENCE INTERVALS:")
    print(f1_df.round(4).to_string(index=False))

    print("\nPAIRED MACRO-F1 DIFFERENCES:")
    print(delta_df.round(4).to_string(index=False))

    print("\nPAIRED MODEL FLIP-RATE DIFFERENCES:")
    print(model_delta_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()