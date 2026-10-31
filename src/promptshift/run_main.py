"""Run the IS6170 main experiment with checkpointing and safe resumption."""

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
REQUESTS_PATH = ROOT / "data" / "main_requests.parquet"
OUTPUT_PATH = ROOT / "data" / "main_predictions.jsonl"

MODELS = ["gemma4-26b", "qwen3.6-36b", "llama-3.3-70b"]

TEMPERATURE = 0
MAX_TOKENS = 16
DELAY_SECONDS = 0.25
SHUFFLE_SEED = 6173


def prompt_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be positive")

    requests = pd.read_parquet(REQUESTS_PATH)

    assert list(requests.columns) == [
        "model", "row_id", "prompt_id", "prompt_text"
    ]
    assert len(requests) == 2700
    assert not requests.duplicated(
        ["model", "row_id", "prompt_id"]
    ).any()
    assert set(requests["model"]) == set(MODELS)
    assert requests.groupby("model").size().eq(900).all()
    assert requests.groupby(
        ["model", "row_id"]
    )["prompt_id"].nunique().eq(3).all()

    expected = {
        (row.model, int(row.row_id), row.prompt_id):
        prompt_hash(row.prompt_text)
        for row in requests.itertuples(index=False)
    }

    completed = set()

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)

                key = (
                    record["model"],
                    int(record["row_id"]),
                    record["prompt_id"],
                )

                assert key in expected, f"Unexpected result: {key}"
                assert key not in completed, f"Duplicate result: {key}"
                assert record["prompt_sha256"] == expected[key]

                completed.add(key)

    # Fixed shuffle avoids evaluating all requests for one model first.
    requests = requests.sample(
        frac=1,
        random_state=SHUFFLE_SEED,
    )

    client = OpenAI(
        base_url="https://maki.uni-mannheim.de/v1",
        api_key=os.environ["MAKI_API_KEY"],
        timeout=60.0,
        max_retries=0,
    )

    calls = 0

    for request in requests.itertuples(index=False):
        model = request.model

        key = (
            model,
            int(request.row_id),
            request.prompt_id,
        )

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
            "prompt_sha256": prompt_hash(request.prompt_text),
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
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        completed.add(key)
        calls += 1

        print(
            "SAVED:",
            model,
            request.prompt_id,
            "row_id=",
            request.row_id,
        )

        if response.model != model:
            raise RuntimeError(
                f"Model mismatch: requested {model}, got {response.model}"
            )

        if calls >= args.limit:
            break

        time.sleep(DELAY_SECONDS)

    print("MAIN_RUN_OK")
    print("New requests:", calls)
    print("Total completed:", len(completed))
    print("Total required:", len(expected))
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()