"""Run and record the IS6170 pilot experiment."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
REQUESTS_PATH = ROOT / "data" / "pilot_requests.parquet"
OUTPUT_PATH = ROOT / "data" / "pilot_predictions.jsonl"

MODELS = [
    "gemma4-26b",
    "qwen3.6-36b",
    "llama-3.3-70b",
]

TEMPERATURE = 0
MAX_TOKENS = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be positive")

    requests = pd.read_parquet(REQUESTS_PATH)

    assert len(requests) == 36
    assert not requests.duplicated(["row_id", "prompt_id"]).any()

    completed = set()

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                completed.add((
                    record["model"],
                    record["row_id"],
                    record["prompt_id"],
                ))

    client = OpenAI(
        base_url="https://maki.uni-mannheim.de/v1",
        api_key=os.environ["MAKI_API_KEY"],
        timeout=60.0,
        max_retries=0,
    )

    calls = 0

    for model in MODELS:
        for request in requests.itertuples(index=False):
            key = (model, int(request.row_id), request.prompt_id)

            if key in completed:
                continue

            start = time.perf_counter()

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": request.prompt_text}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            elapsed = time.perf_counter() - start

            record = {
                "row_id": int(request.row_id),
                "prompt_id": request.prompt_id,
                "model": model,
                "response_model": response.model,
                "raw_response": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "latency_seconds": round(elapsed, 3),
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            with OUTPUT_PATH.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")

            print(
                "SAVED:",
                model,
                request.prompt_id,
                "row_id=",
                request.row_id,
            )

            calls += 1

            if calls >= args.limit:
                print("PILOT_RUN_OK")
                print("New requests:", calls)
                return

    print("PILOT_RUN_OK")
    print("New requests:", calls)
    print("All available requests completed.")


if __name__ == "__main__":
    main()