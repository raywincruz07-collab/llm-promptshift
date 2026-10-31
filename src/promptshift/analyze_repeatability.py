"""Analyze repeated LLM classifications for IS6170."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

SELECTED_POSTS = [193, 519, 929, 1247]


def main():
    original = pd.read_json(
        DATA / "pilot_predictions.jsonl",
        lines=True,
    )

    original = original[
        original["row_id"].isin(SELECTED_POSTS)
    ].copy()

    original["repetition"] = 1

    repeats = pd.read_json(
        DATA / "repeatability_predictions.jsonl",
        lines=True,
    )

    combined = pd.concat(
        [original, repeats],
        ignore_index=True,
    )

    keys = ["model", "row_id", "prompt_id"]

    assert len(combined) == 108

    assert not combined.duplicated(
        keys + ["repetition"]
    ).any()

    assert combined.groupby(keys).size().eq(3).all()

    table = combined.pivot(
        index=keys,
        columns="repetition",
        values="raw_response",
    )

    assert set(table.columns) == {1, 2, 3}

    table = table.rename(columns={
        1: "original",
        2: "repeat_2",
        3: "repeat_3",
    })

    table["stable"] = table[
        ["original", "repeat_2", "repeat_3"]
    ].nunique(axis=1).eq(1)

    summary = table.groupby(level="model")[
        "stable"
    ].agg(["sum", "count"])

    summary = summary.rename(columns={
        "sum": "stable_cells",
        "count": "total_cells",
    })

    summary["unstable_cells"] = (
        summary["total_cells"] - summary["stable_cells"]
    )

    table.reset_index().to_csv(
        DATA / "repeatability_agreement.csv",
        index=False,
    )

    summary.to_csv(
        DATA / "repeatability_summary.csv",
    )

    print("REPEATABILITY_ANALYSIS_OK")

    print("\nMODEL SUMMARY:")
    print(summary.to_string())

    print("\nUNSTABLE COMBINATIONS:")
    unstable = table[~table["stable"]]

    if unstable.empty:
        print("None")
    else:
        print(unstable.to_string())


if __name__ == "__main__":
    main()