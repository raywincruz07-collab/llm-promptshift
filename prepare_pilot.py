
"""Prepare TweetEval pilot requests for the IS6170 experiment."""

from pathlib import Path

import pandas as pd

from prompts import PROMPTS, render_prompt


PROJECT_DIR = Path(__file__).resolve().parent
PILOT_PATH = PROJECT_DIR / "data" / "pilot12.parquet"
OUTPUT_PATH = PROJECT_DIR / "data" / "pilot_requests.parquet"


def main():
    # Step 1: Load the 12 selected social-media posts.
    pilot = pd.read_parquet(PILOT_PATH)

    # Step 2: Verify the pilot dataset.
    assert len(pilot) == 12
    assert pilot["text"].is_unique
    assert pilot["label"].value_counts().to_dict() == {0: 6, 1: 6}

    # Step 3: Prepare each post with all three prompts.
    requests = []

    for row in pilot.itertuples(index=False):
        for prompt_id in PROMPTS:
            requests.append(
                {
                    "row_id": int(row.row_id),
                    "prompt_id": prompt_id,
                    "prompt_text": render_prompt(prompt_id, row.text),
                }
            )

    # Step 4: Save the requests.
    request_df = pd.DataFrame(requests)

    assert len(request_df) == 36
    assert not request_df.duplicated(["row_id", "prompt_id"]).any()

    request_df.to_parquet(OUTPUT_PATH, index=False)

    print("PILOT_PREPARATION_OK")
    print("Posts:", len(pilot))
    print("Prompts:", len(PROMPTS))
    print("Prepared requests:", len(request_df))
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
