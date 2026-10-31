"""Prepare repeatability requests for the IS6170 pilot."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

INPUT = DATA / "pilot_requests.parquet"
OUTPUT = DATA / "repeatability_requests.parquet"

SELECTED_POSTS = [193, 519, 929, 1247]

MODELS = [
    "gemma4-26b",
    "qwen3.6-36b",
    "llama-3.3-70b",
]

REPETITIONS = [2, 3]


def main():
    original = pd.read_parquet(INPUT)

    selected = original[
        original["row_id"].isin(SELECTED_POSTS)
    ].copy()

    assert len(selected) == 12
    assert selected["row_id"].nunique() == 4

    assert selected.groupby("row_id")[
        "prompt_id"
    ].nunique().eq(3).all()

    requests = []

    for repetition in REPETITIONS:
        for model in MODELS:
            for row in selected.itertuples(index=False):
                requests.append({
                    "repetition": repetition,
                    "model": model,
                    "row_id": int(row.row_id),
                    "prompt_id": row.prompt_id,
                    "prompt_text": row.prompt_text,
                })

    request_df = pd.DataFrame(requests)

    assert len(request_df) == 72

    assert not request_df.duplicated([
        "repetition",
        "model",
        "row_id",
        "prompt_id",
    ]).any()

    request_df.to_parquet(OUTPUT, index=False)

    print("REPEATABILITY_PREPARATION_OK")
    print("Selected posts:", len(SELECTED_POSTS))
    print("Models:", len(MODELS))
    print("Prompts:", 3)
    print("Additional repetitions:", len(REPETITIONS))
    print("Prepared requests:", len(request_df))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()