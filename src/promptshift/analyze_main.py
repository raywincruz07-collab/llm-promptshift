"""Analyze the IS6170 main prompt-robustness experiment."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS = ROOT / "data" / "main_predictions.jsonl"
DATASET = ROOT / "data" / "main300.parquet"

METRICS_OUTPUT = ROOT / "data" / "main_metrics.csv"
FLIPS_OUTPUT = ROOT / "data" / "main_flips.csv"

MODELS = [
    "gemma4-26b",
    "qwen3.6-36b",
    "llama-3.3-70b",
]

PROMPTS = ["P0", "P1", "P2"]

LABEL_MAP = {
    "NOT_OFFENSIVE": 0,
    "OFFENSIVE": 1,
}


def main():
    predictions = pd.read_json(PREDICTIONS, lines=True)
    dataset = pd.read_parquet(DATASET)

    # Verify the experiment is complete.
    assert len(predictions) == 2700
    assert len(dataset) == 300

    assert set(predictions["model"]) == set(MODELS)
    assert set(predictions["prompt_id"]) == set(PROMPTS)

    assert not predictions.duplicated(
        ["model", "row_id", "prompt_id"]
    ).any()

    assert predictions["finish_reason"].eq("stop").all()
    assert predictions["model"].eq(
        predictions["response_model"]
    ).all()

    predictions["predicted_label"] = (
        predictions["raw_response"].map(LABEL_MAP)
    )

    assert predictions["predicted_label"].notna().all()

    # Attach the dataset's gold-standard labels.
    results = predictions.merge(
        dataset[["row_id", "label"]],
        on="row_id",
        validate="many_to_one",
    )

    assert len(results) == 2700
    assert results["label"].notna().all()

    metrics = []
    flips = []

    for model in MODELS:
        model_data = results[
            results["model"] == model
        ]

        assert len(model_data) == 900

        # Compare predictions across all three prompts.
        wide = model_data.pivot(
            index="row_id",
            columns="prompt_id",
            values="predicted_label",
        )

        wide = wide[PROMPTS]

        assert wide.shape == (300, 3)
        assert not wide.isna().any().any()

        any_flip = wide.nunique(axis=1).gt(1)

        p0_p1 = (wide["P0"] != wide["P1"])
        p0_p2 = (wide["P0"] != wide["P2"])
        p1_p2 = (wide["P1"] != wide["P2"])

        flips.append({
            "model": model,
            "total_posts": 300,
            "posts_with_any_flip": int(any_flip.sum()),
            "any_flip_rate": any_flip.mean(),
            "p0_vs_p1_flips": int(p0_p1.sum()),
            "p0_vs_p2_flips": int(p0_p2.sum()),
            "p1_vs_p2_flips": int(p1_p2.sum()),
        })

        # Calculate classification performance for each prompt.
        for prompt_id in PROMPTS:
            subset = model_data[
                model_data["prompt_id"] == prompt_id
            ]

            assert len(subset) == 300

            y_true = subset["label"]
            y_pred = subset["predicted_label"]

            tn, fp, fn, tp = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1],
            ).ravel()

            metrics.append({
                "model": model,
                "prompt_id": prompt_id,
                "accuracy": accuracy_score(
                    y_true,
                    y_pred,
                ),
                "macro_f1": f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    labels=[0, 1],
                    zero_division=0,
                ),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "offensive_recall": (
                    tp / (tp + fn)
                ),
                "nonoffensive_recall": (
                    tn / (tn + fp)
                ),
                "median_latency_seconds": subset[
                    "latency_seconds"
                ].median(),
            })

    metrics_df = pd.DataFrame(metrics)
    flips_df = pd.DataFrame(flips)

    # Save derived analysis without modifying predictions.
    metrics_df.to_csv(METRICS_OUTPUT, index=False)
    flips_df.to_csv(FLIPS_OUTPUT, index=False)

    print("MAIN_ANALYSIS_OK")

    print("\nMODEL AND PROMPT METRICS:")
    print(metrics_df.round(3).to_string(index=False))

    print("\nPROMPT FLIP ANALYSIS:")
    print(flips_df.round(3).to_string(index=False))

    print("\nOUTPUT FILES:")
    print(METRICS_OUTPUT)
    print(FLIPS_OUTPUT)


if __name__ == "__main__":
    main()