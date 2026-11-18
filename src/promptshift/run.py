"""Send a stage's requests (or the probe) to the API, resumably.

    python src/promptshift/run.py --stage pilot --all
    python src/promptshift/run.py --stage main --limit 20   # live smoke test
    python src/promptshift/run.py --stage main --all        # full run (resumable)
    python src/promptshift/run.py --stage main --all --probe

Rules (from the protocol):
- One request in flight at a time; fixed delay between calls.
- Valid answers are never re-sent, even if the label looks wrong.
- Only transport failures are retried; every attempt is logged.
- Non-compliant answers are kept (compliant = false), never dropped.
- Labels are not printed while running, so nobody inspects results early.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
from api import (  # noqa: E402
    ModelMismatchError, StopRuleTriggered, call, is_transport_error,
    make_client, response_record,
)
from common import append_jsonl, read_json, read_jsonl, sha256_file, utc_now  # noqa: E402


def load_done(output_path, requests_by_id):
    done = {}
    for record in read_jsonl(output_path):
        request_id = record["request_id"]
        assert request_id in requests_by_id, f"unexpected result {request_id}"
        assert request_id not in done, f"duplicate result {request_id}"
        assert record["prompt_sha256"] == requests_by_id[request_id]["prompt_sha256"]
        done[request_id] = record
    return done


def run(manifest_path, output_path, attempts_path, client, limit=None,
        delay=C.DELAY_SECONDS, backoff_base=2.0, request_logprobs=C.REQUEST_LOGPROBS,
        max_retries=C.MAX_TRANSPORT_RETRIES,
        stop_after=C.STOP_AFTER_CONSECUTIVE_TRANSPORT_FAILURES,
        expected_manifest_sha256=None, quiet=False):
    if expected_manifest_sha256 is not None:
        actual = sha256_file(manifest_path)
        if actual != expected_manifest_sha256:
            raise SystemExit("requests file changed since prepare step; refusing to run")

    requests = sorted(read_jsonl(manifest_path), key=lambda r: r["order"])
    by_id = {r["request_id"]: r for r in requests}
    done = load_done(output_path, by_id)
    pending = [r for r in requests if r["request_id"] not in done]

    completed_now = 0
    consecutive_failures = 0
    for request in pending:
        if limit is not None and completed_now >= limit:
            break
        attempt = 0
        while True:
            attempt += 1
            start = time.perf_counter()
            try:
                response = call(client, request, request_logprobs)
            except Exception as error:  # noqa: BLE001 - classified below
                transport = is_transport_error(error)
                append_jsonl(attempts_path, {
                    "request_id": request["request_id"], "attempt": attempt,
                    "timestamp_utc": utc_now(), "ok": False, "transport": transport,
                    "error_type": type(error).__name__, "error": str(error)[:500],
                    "latency_seconds": round(time.perf_counter() - start, 3),
                })
                if not transport:
                    raise  # bad request, auth, unknown model: fix config, then resume
                if attempt <= max_retries:
                    time.sleep(backoff_base * 2 ** (attempt - 1))
                    continue
                consecutive_failures += 1
                if consecutive_failures >= stop_after:
                    raise StopRuleTriggered(
                        f"{consecutive_failures} requests in a row failed at transport level; "
                        "stopped. Check the API, then rerun to resume."
                    ) from error
                break  # leave this request pending; a later run retries it
            latency = time.perf_counter() - start
            record = response_record(request, response, latency, attempt, request_logprobs)
            append_jsonl(attempts_path, {
                "request_id": request["request_id"], "attempt": attempt,
                "timestamp_utc": record["timestamp_utc"], "ok": True,
                "transport": False, "latency_seconds": record["latency_seconds"],
            })
            if record["response_model"] != request["model"]:
                raise ModelMismatchError(
                    f"asked for {request['model']}, got {record['response_model']}; stopped"
                )
            append_jsonl(output_path, record)
            consecutive_failures = 0
            completed_now += 1
            break
        if not quiet and completed_now and completed_now % 100 == 0:
            print(f"progress: {len(done) + completed_now}/{len(requests)}", flush=True)
        if delay:
            time.sleep(delay)

    final = load_done(output_path, by_id)
    summary = {
        "total": len(requests),
        "completed": len(final),
        "remaining": len(requests) - len(final),
        "completed_this_session": completed_now,
        "non_compliant_so_far": sum(not r["compliant"] for r in final.values()),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--limit", type=int, help="send at most this many requests")
    group.add_argument("--all", action="store_true", help="send every pending request")
    parser.add_argument("--stage", choices=sorted(C.STAGES), required=True)
    parser.add_argument("--probe", action="store_true", help="run the contamination probe")
    parser.add_argument("--mock", action="store_true", help="offline mock API (dry run)")
    args = parser.parse_args()

    data_dir = C.stage_dir(args.stage)
    if args.probe:
        manifest = data_dir / "probe_requests.jsonl"
        output = data_dir / "probe_predictions.jsonl"
        attempts = data_dir / "probe_attempts.jsonl"
        expected = read_json(data_dir / "probe_meta.json")["requests_file_sha256"]
    else:
        manifest = data_dir / "requests.jsonl"
        output = data_dir / "predictions.jsonl"
        attempts = data_dir / "attempts.jsonl"
        expected = read_json(data_dir / "manifest_meta.json")["requests_file_sha256"]

    client = make_client(mock=args.mock)
    summary = run(
        manifest, output, attempts, client,
        limit=None if args.all else args.limit,
        delay=0 if args.mock else C.DELAY_SECONDS,
        backoff_base=0 if args.mock else 2.0,
        expected_manifest_sha256=expected,
    )
    print("RUN_SESSION_DONE", summary)
    if summary["remaining"] == 0:
        print("ALL_REQUESTS_COMPLETE")


if __name__ == "__main__":
    main()
