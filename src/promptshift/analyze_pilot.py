"""Analyze the IS6170 pilot experiment."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS = ROOT / "data" / "pilot_predictions.jsonl"
PILOT = ROOT / "data" / "pilot12.parquet"

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
    pilot = pd.read_parquet(PILOT)

    # Check the experimental data.
    assert len(predictions) == 108
    assert len(pilot) == 12

    assert set(predictions["model"]) == set(MODELS)
    assert set(predictions["prompt_id"]) == set(PROMPTS)

    assert not predictions.duplicated(
        ["model", "row_id", "prompt_id"]
    ).any()

    assert predictions["finish_reason"].eq("stop").all()

    predictions["predicted_label"] = (
        predictions["raw_response"].map(LABEL_MAP)
    )

    assert predictions["predicted_label"].notna().all()

    # Attach the correct dataset labels.
    results = predictions.merge(
        pilot[["row_id", "label"]],
        on="row_id",
        validate="many_to_one",
    )

    assert len(results) == 108
    assert results["label"].notna().all()

    metrics = []
    flips = []

    for model in MODELS:
        model_data = results[results["model"] == model]

        assert len(model_data) == 36

        # Compare predictions across prompts.
        wide = model_data.pivot(
            index="row_id",
            columns="prompt_id",
            values="predicted_label",
        )

        assert wide.shape == (12, 3)

        flips.append({
            "model": model,
            "posts_with_any_flip": int(
                wide.nunique(axis=1).gt(1).sum()
            ),
            "total_posts": len(wide),
            "p0_vs_p1_flips": int(
                (wide["P0"] != wide["P1"]).sum()
            ),
            "p0_vs_p2_flips": int(
                (wide["P0"] != wide["P2"]).sum()
            ),
        })

        for prompt_id in PROMPTS:
            subset = model_data[
                model_data["prompt_id"] == prompt_id
            ]

            assert len(subset) == 12

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
                "accuracy": accuracy_score(y_true, y_pred),
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
                "median_latency_seconds": subset[
                    "latency_seconds"
                ].median(),
            })

    metrics_df = pd.DataFrame(metrics)
    flips_df = pd.DataFrame(flips)

    metrics_df.to_csv(
        ROOT / "data" / "pilot_metrics.csv",
        index=False,
    )

    flips_df.to_csv(
        ROOT / "data" / "pilot_flips.csv",
        index=False,
    )

    print("PILOT_ANALYSIS_OK")

    print("\nMODEL AND PROMPT METRICS:")
    print(metrics_df.round(3).to_string(index=False))

    print("\nPROMPT FLIP ANALYSIS:")
    print(flips_df.to_string(index=False))


if __name__ == "__main__":
    main()