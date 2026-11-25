"""The fresh stages build the same 8-condition design from Bluesky posts (no labels needed yet)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "promptshift"))

import config as C  # noqa: E402
import prepare as P  # noqa: E402
from common import read_jsonl, write_jsonl  # noqa: E402
from mock_api import MockClient  # noqa: E402
from run import run  # noqa: E402


def test_fresh_stage_prepare_and_mock_run(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DATA_DIR", tmp_path)
    posts = [{"row_id": f"main{i:04d}", "text": f"@USER you are post {i}", "keyword": "you are",
              "window": "2026-05-01", "created_at": "2026-05-02T00:00:00Z", "author_hash": "x"} for i in range(50)]
    write_jsonl(tmp_path / "fresh" / "sample_main.jsonl", posts)
    records = P.load_fresh("fresh")
    assert set(records[0]) == {"row_id", "text"}            # nothing else leaks into requests
    fp = P.prepare(records, "fresh", tmp_path / "fresh_stage", tmp_path / "protocol")
    assert fp["n_requests"] == 50 * len(C.CONDITIONS) * len(C.MODELS)
    assert fp["label_counts"] == {"unlabelled_at_prepare": 50}
    assert fp["dataset"]["source"].startswith("bluesky")
    requests = read_jsonl(tmp_path / "fresh_stage" / "requests.jsonl")
    assert P.adjacency_violations(requests) == 0
    summary = run(tmp_path / "fresh_stage" / "requests.jsonl", tmp_path / "fresh_stage" / "predictions.jsonl",
                  tmp_path / "fresh_stage" / "attempts.jsonl", MockClient(), delay=0, backoff_base=0,
                  request_logprobs=False, quiet=True, expected_manifest_sha256=fp["requests_file_sha256"])
    assert summary["remaining"] == 0
