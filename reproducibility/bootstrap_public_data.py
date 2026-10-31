"""Rebuild experimental datasets and requests from public source + manifests.

Does not read the project's original data/ directory.
Does not contact the model API.
"""

from pathlib import Path
from urllib.request import urlopen
import hashlib
import json
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPRO = ROOT / "reproducibility"
CACHE = REPRO / "source_cache"
OUTPUT = REPRO / "fresh_build"

sys.path.insert(0, str(ROOT / "src" / "promptshift"))
from prompts import PROMPTS, render_prompt

REPOSITORY = "cardiffnlp/tweet_eval"
REVISION = "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"

MODELS = [
    "gemma4-26b",
    "qwen3.6-36b",
    "llama-3.3-70b",
]


def load_json(name):
    return json.loads(
        (REPRO / name).read_text(encoding="utf-8")
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(split, expected_hash):
    filename = f"{split}-00000-of-00001.parquet"
    path = CACHE / filename

    CACHE.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        url = (
            f"https://huggingface.co/datasets/{REPOSITORY}"
            f"/resolve/{REVISION}/offensive/{filename}"
        )

        print(f"Downloading {split}...", flush=True)

        temporary = path.with_suffix(".part")

        try:
            with urlopen(url, timeout=60) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)

            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    assert sha256(path) == expected_hash, (
        f"{split}: downloaded source hash mismatch"
    )

    return pd.read_parquet(path)


def save_verified(df, filename, expected_hash=None):
    path = OUTPUT / filename

    if not path.exists():
        df.to_parquet(path, index=False)

    saved = pd.read_parquet(path)

    pd.testing.assert_frame_equal(
        saved, df, check_exact=True
    )

    if expected_hash is not None:
        assert sha256(path) == expected_hash, (
            f"{filename}: Parquet byte hash differs. "
            "Check package versions and serialization."
        )

    print(f"VERIFIED: {filename} ({len(saved)} rows)")
    return saved


def make_requests(sample, include_models):
    records = []

    models = MODELS if include_models else [None]

    for model in models:
        for row in sample.itertuples(index=False):
            for prompt_id in PROMPTS:
                record = {}

                if include_models:
                    record["model"] = model

                record["row_id"] = int(row.row_id)
                record["prompt_id"] = prompt_id
                record["prompt_text"] = render_prompt(
                    prompt_id, row.text
                )

                records.append(record)

    return pd.DataFrame(records)


def main():
    samples = load_json("sample_manifest_v1.json")
    suffix_info = load_json("test_text_suffix_manifest.json")
    sources = load_json("tweeteval_source_manifest.json")

    assert sources["revision"] == REVISION
    assert suffix_info["source_revision"] == REVISION
    assert list(PROMPTS) == ["P0", "P1", "P2"]

    OUTPUT.mkdir(parents=True, exist_ok=True)

    validation = load_source(
        "validation",
        sources["splits"]["validation"]["source_sha256"],
    )

    test = load_source(
        "test",
        sources["splits"]["test"]["source_sha256"],
    )

    assert list(validation.columns) == ["text", "label"]
    assert {"text", "label"}.issubset(test.columns)

    assert len(validation) == 1324
    assert len(test) == 860

    assert validation["label"].value_counts().to_dict() == {
        0: 865, 1: 459
    }
    assert test["label"].value_counts().to_dict() == {
        0: 620, 1: 240
    }

    validation = save_verified(
        validation[["text", "label"]],
        "validation.parquet",
    )

    rebuilt_samples = {}

    for name, filename in [
        ("pilot", "pilot12.parquet"),
        ("main", "main300.parquet"),
    ]:
        info = samples[name]
        ids = info["row_ids_in_original_order"]

        assert len(ids) == info["rows"]
        assert len(set(ids)) == len(ids)
        assert all(0 <= i < len(validation) for i in ids)

        rebuilt = validation.iloc[ids].reset_index(
            drop=True
        ).copy()

        rebuilt.insert(0, "row_id", ids)

        rebuilt_samples[name] = save_verified(
            rebuilt,
            filename,
            expected_hash=info["sha256"],
        )

    pilot = rebuilt_samples["pilot"]
    main = rebuilt_samples["main"]

    assert set(pilot["row_id"]).isdisjoint(
        set(main["row_id"])
    )

    assert pilot["label"].value_counts().to_dict() == {
        0: 6, 1: 6
    }

    assert main["label"].value_counts().to_dict() == {
        0: 196, 1: 104
    }

    suffixes = suffix_info["suffixes_in_row_order"]

    assert len(suffixes) == 860
    assert all(
        isinstance(s, str) and s.isspace()
        for s in suffixes
    )

    rebuilt_test = test[["text", "label"]].copy()

    rebuilt_test["text"] = [
        text + suffix
        for text, suffix in zip(
            rebuilt_test["text"], suffixes
        )
    ]

    rebuilt_test.insert(0, "row_id", range(860))

    rebuilt_test = save_verified(
        rebuilt_test,
        "test860.parquet",
        expected_hash=suffix_info[
            "original_experiment_sha256"
        ],
    )

    total_requests = 0

    for name, sample, include_models, expected in [
        ("pilot", pilot, False, 36),
        ("main", main, True, 2700),
        ("test", rebuilt_test, True, 7740),
    ]:
        requests = make_requests(
            sample, include_models
        )

        assert len(requests) == expected

        keys = ["row_id", "prompt_id"]

        if include_models:
            keys.insert(0, "model")

        assert not requests.duplicated(keys).any()

        save_verified(
            requests,
            f"{name}_requests.parquet",
        )

        total_requests += len(requests)

    assert total_requests == 10476

    print()
    print("PUBLIC_DATA_BOOTSTRAP_OK")
    print("Original historical samples reconstructed: 312")
    print("Original test posts reconstructed: 860")
    print("Prepared requests reconstructed:", total_requests)
    print("Original data/ directory accessed: NO")
    print("LLM API calls: 0")
    print("Output:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
