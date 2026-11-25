"""Download the pinned dataset and build a stage's sample and request list.

    python src/promptshift/prepare.py --stage pilot
    python src/promptshift/prepare.py --stage main

Makes no model calls. Post text is written only under data/ (gitignored);
hash-only fingerprints go to protocol/ for the registration.
"""

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
from common import (  # noqa: E402
    git_state, read_jsonl, sha256_file, sha256_text, utc_now, write_json, write_jsonl,
)
from prompts import prompt_registry, render  # noqa: E402


# ----------------------------------------------------------------------- data
def download_split(split: str, cache_dir: Path = C.CACHE_DIR) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{split}-00000-of-00001.parquet"
    if not path.exists():
        url = (f"https://huggingface.co/datasets/{C.DATASET_REPO}/resolve/"
               f"{C.DATASET_REVISION}/offensive/{path.name}")
        temporary = path.with_suffix(".part")
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    assert sha256_file(path) == C.SPLIT_SHA256[split], f"{split}: file hash mismatch"
    return path


def load_split(split: str) -> list:
    import pandas as pd

    frame = pd.read_parquet(download_split(split))
    records = [
        {"row_id": i, "label": int(row.label), "text": row.text}
        for i, row in enumerate(frame.itertuples(index=False))
    ]
    counts = dict(Counter(r["label"] for r in records))
    assert counts == C.SPLIT_LABEL_COUNTS[split], (split, counts)
    return records


def load_fresh(stage: str) -> list:
    """Fresh posts (no gold labels needed to build requests; labels are joined at analysis)."""
    path = C.DATA_DIR / "fresh" / C.STAGES[stage]["file"]
    records = read_jsonl(path)
    if not records:
        raise SystemExit(f"{path} not found: run collect_fresh.py sample first.")
    return [{"row_id": r["row_id"], "text": r["text"]} for r in records]


def stage_sample(stage: str, records: list) -> list:
    n = C.STAGES[stage].get("n_posts")
    if n is None:
        return records
    rng = np.random.default_rng(C.PILOT_SEED)
    chosen = sorted(rng.choice(len(records), size=n, replace=False))
    return [records[i] for i in chosen]


# ----------------------------------------------------------------- manifest
def build_requests(records, stage, models=C.MODELS, conditions=C.CONDITIONS):
    requests = []
    for model in models:
        for record in records:
            for condition in conditions:
                prompt = render(condition, record["text"])
                requests.append({
                    "request_id": f"{stage}:{model}:{record['row_id']}:{condition}",
                    "task": "classify",
                    "sample": stage,
                    "model": model,
                    "row_id": record["row_id"],
                    "condition": condition,
                    "max_tokens": C.MAX_TOKENS_CLASSIFY,
                    "prompt_text": prompt,
                    "prompt_sha256": sha256_text(prompt),
                })
    return requests


def adjacency_violations(ordered):
    """Repeat pairs (W0/R0, D0/RD0; same model and post) sent back to back."""
    position = {(r["model"], r["row_id"], r["condition"]): i for i, r in enumerate(ordered)}
    violations = 0
    for (model, row_id, condition), i in position.items():
        if condition in C.REPEAT_OF:
            j = position.get((model, row_id, C.REPEAT_OF[condition]))
            if j is not None and abs(i - j) == 1:
                violations += 1
    return violations


def shuffle_requests(requests, seed, max_attempts=1000):
    for attempt in range(max_attempts):
        rng = np.random.default_rng([seed, attempt])
        ordered = [requests[i] for i in rng.permutation(len(requests))]
        if adjacency_violations(ordered) == 0:
            for index, request in enumerate(ordered):
                request["order"] = index
            return ordered, attempt
    raise RuntimeError("could not satisfy the adjacency rule")


def prepare(records, stage, data_dir, protocol_dir, seed=C.SHUFFLE_SEED, force=False):
    data_dir, protocol_dir = Path(data_dir), Path(protocol_dir)
    predictions = data_dir / "predictions.jsonl"
    if predictions.exists() and read_jsonl(predictions) and not force:
        raise SystemExit("predictions.jsonl already has results; refusing to rebuild this stage.")

    registry_path = protocol_dir / "prompt_registry.json"
    registry = prompt_registry()
    write_json(registry_path, registry)
    write_jsonl(data_dir / "sample.jsonl", records)
    ordered, attempts = shuffle_requests(build_requests(records, stage), seed)
    write_jsonl(data_dir / "requests.jsonl", ordered)

    counts = Counter(r.get("label") for r in records)
    source = C.STAGES.get(stage, {}).get("source")
    fingerprint = {
        "created_utc": utc_now(),
        "git": git_state(C.ROOT),
        "stage": stage,
        "dataset": ({"source": "bluesky (collect_fresh.py)",
                     "file": C.STAGES[stage]["file"],
                     "file_sha256": sha256_file(C.DATA_DIR / "fresh" / C.STAGES[stage]["file"])}
                    if source == "fresh" else
                    {"repo": C.DATASET_REPO, "revision": C.DATASET_REVISION,
                     "split": C.STAGES.get(stage, {}).get("split")}),
        "n_posts": len(records),
        "label_counts": ({"unlabelled_at_prepare": len(records)} if source == "fresh" else
                         {"NOT_OFFENSIVE": counts.get(0, 0), "OFFENSIVE": counts.get(1, 0)}),
        "sample_row_ids_sha256": sha256_text(",".join(str(r["row_id"]) for r in records)),
        "models": C.MODELS,
        "conditions": C.CONDITIONS,
        "n_requests": len(ordered),
        "shuffle_seed": seed,
        "shuffle_attempts_needed": attempts + 1,
        "requests_file_sha256": sha256_file(data_dir / "requests.jsonl"),
        "prompt_registry_sha256": sha256_file(registry_path),
        "template_sha256": {c: v["template_sha256"] for c, v in registry["conditions"].items()},
        "inference": {"temperature": C.TEMPERATURE, "max_tokens": C.MAX_TOKENS_CLASSIFY,
                      "request_logprobs": C.REQUEST_LOGPROBS, "concurrency": 1},
        "primary_h2": C.PRIMARY_H2,
    }
    write_json(data_dir / "manifest_meta.json", fingerprint)
    write_json(protocol_dir / f"{stage}_fingerprint.json", fingerprint)
    return fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(C.STAGES), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if C.STAGES[args.stage].get("source") == "fresh":
        records = load_fresh(args.stage)
    else:
        records = stage_sample(args.stage, load_split(C.STAGES[args.stage]["split"]))
    fingerprint = prepare(records, args.stage, C.stage_dir(args.stage), C.PROTOCOL_DIR, force=args.force)
    print("PREPARE_OK")
    for key in ("stage", "n_posts", "label_counts", "n_requests", "requests_file_sha256",
                "prompt_registry_sha256", "git"):
        print(f"{key}: {fingerprint[key]}")
    if fingerprint["git"]["dirty"]:
        print("WARNING: uncommitted changes. Commit before registering.")


if __name__ == "__main__":
    main()
