"""Run the IS6170 repeatability experiment."""

import argparse
import hashlib
import json
import os
import time

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

INPUT = DATA / "repeatability_requests.parquet"
OUTPUT = DATA / "repeatability_predictions.jsonl"

TEMPERATURE = 0
MAX_TOKENS = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be positive")

    requests = pd.read_parquet(INPUT)

    assert len(requests) == 72

    assert not requests.duplicated([
        "repetition",
        "model",
        "row_id",
        "prompt_id",
    ]).any()

    # Identify requests already completed.
    completed = set()

    if OUTPUT.exists():
        with OUTPUT.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)

                completed.add((
                    int(record["repetition"]),
                    record["model"],
                    int(record["row_id"]),
                    record["prompt_id"],
                ))

    client = OpenAI(
        base_url="https://maki.uni-mannheim.de/v1",
        api_key=os.environ["MAKI_API_KEY"],
        timeout=60.0,
        max_retries=0,
    )

    calls = 0

    for request in requests.itertuples(index=False):
        key = (
            int(request.repetition),
            request.model,
            int(request.row_id),
            request.prompt_id,
        )

        if key in completed:
            continue

        start = time.perf_counter()

        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {
                    "role": "user",
                    "content": request.prompt_text,
                }
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        elapsed = time.perf_counter() - start

        prompt_hash = hashlib.sha256(
            request.prompt_text.encode("utf-8")
        ).hexdigest()

        record = {
            "repetition": int(request.repetition),
            "model": request.model,
            "response_model": response.model,
            "row_id": int(request.row_id),
            "prompt_id": request.prompt_id,
            "prompt_sha256": prompt_hash,
            "raw_response": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "latency_seconds": round(elapsed, 3),
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        # Save each successful response immediately.
        with OUTPUT.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        calls += 1

        print(
            "SAVED:",
            "repeat=",
            request.repetition,
            request.model,
            request.prompt_id,
            "row_id=",
            request.row_id,
        )

        if calls >= args.limit:
            break

    print("REPEATABILITY_RUN_OK")
    print("New requests:", calls)
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()