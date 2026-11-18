"""Pipeline tests. Run with: python -m pytest tests -q

No network and no real posts are needed.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "promptshift"))

import config as C  # noqa: E402
import stats as S  # noqa: E402
from analyze import analyse  # noqa: E402
from api import ModelMismatchError  # noqa: E402
from common import read_jsonl, sha256_text  # noqa: E402
from mock_api import MockClient  # noqa: E402
from prepare import adjacency_violations, build_requests, prepare, shuffle_requests  # noqa: E402
from probe import rouge_l_f1, split_post  # noqa: E402
from prompts import BASE_PROMPTS, DEFINITION_BLOCK, condition_templates, render  # noqa: E402
from run import run  # noqa: E402


def posts(n=40, seed=1):
    rng = random.Random(seed)
    return [{"row_id": i, "label": int(rng.random() < 0.3),
             "text": f"post {i} @USER " + " ".join(rng.choice("abcdefg") for _ in range(6))}
            for i in range(n)]


# ------------------------------------------------------------------ prompts
def test_wordings_differ_only_in_the_first_sentence():
    rests = {p: t.split("\n\n", 1)[1] for p, t in BASE_PROMPTS.items()}
    assert len(set(rests.values())) == 1
    assert len({t.split("\n\n", 1)[0] for t in BASE_PROMPTS.values()}) == 3


def test_definition_is_the_only_difference():
    templates = condition_templates()
    for w, d in (("W0", "D0"), ("W1", "D1"), ("W2", "D2")):
        assert DEFINITION_BLOCK in templates[d] and DEFINITION_BLOCK not in templates[w]
        assert templates[d].replace(DEFINITION_BLOCK + "\n\n", "") == templates[w]


def test_repeat_conditions_are_exact_copies():
    assert render("R0", "abc") == render("W0", "abc")
    assert render("RD0", "abc") == render("D0", "abc")


# ----------------------------------------------------------------- manifest
def test_manifest_counts_ids_and_adjacency():
    requests = build_requests(posts(), "main")
    assert len(requests) == 40 * 3 * 8
    assert len({r["request_id"] for r in requests}) == len(requests)
    ordered, _ = shuffle_requests(requests, seed=5)
    assert adjacency_violations(ordered) == 0
    assert [r["order"] for r in ordered] == list(range(len(ordered)))


def test_shuffle_is_deterministic():
    a, _ = shuffle_requests(build_requests(posts(), "main"), seed=9)
    b, _ = shuffle_requests(build_requests(posts(), "main"), seed=9)
    assert [r["request_id"] for r in a] == [r["request_id"] for r in b]


def test_prepare_refuses_to_overwrite_started_study(tmp_path):
    prepare(posts(), "main", tmp_path / "d", tmp_path / "p")
    (tmp_path / "d" / "predictions.jsonl").write_text('{"request_id": "x"}\n')
    with pytest.raises(SystemExit):
        prepare(posts(), "main", tmp_path / "d", tmp_path / "p")


# ------------------------------------------------------------------- runner
def _prepared(tmp_path, n=30):
    records = posts(n)
    meta = prepare(records, "main", tmp_path / "d", tmp_path / "p")
    return records, meta, tmp_path / "d"


def _run(d, client, meta, **kwargs):
    return run(d / "requests.jsonl", d / "predictions.jsonl", d / "attempts.jsonl", client,
               delay=0, backoff_base=0, expected_manifest_sha256=meta["requests_file_sha256"],
               quiet=True, **kwargs)


def test_runner_resumes_retries_and_never_duplicates(tmp_path):
    records, meta, d = _prepared(tmp_path)
    client = MockClient(truth={r["text"]: r["label"] for r in records}, transport_failure=0.1, seed=2)
    first = _run(d, client, meta, limit=100)
    assert first["completed"] == 100
    final = _run(d, client, meta)
    assert final["remaining"] == 0
    ids = [r["request_id"] for r in read_jsonl(d / "predictions.jsonl")]
    assert len(ids) == len(set(ids)) == meta["n_requests"]
    attempts = read_jsonl(d / "attempts.jsonl")
    assert any(not a["ok"] and a["transport"] for a in attempts)


def test_noncompliant_answers_are_kept(tmp_path):
    records, meta, d = _prepared(tmp_path)
    _run(d, MockClient(noncompliance=0.2, seed=3), meta)
    predictions = read_jsonl(d / "predictions.jsonl")
    assert len(predictions) == meta["n_requests"]
    assert any(not p["compliant"] for p in predictions)
    assert all(p["parsed_label"] is None for p in predictions if not p["compliant"])


def test_runner_refuses_changed_manifest(tmp_path):
    records, meta, d = _prepared(tmp_path)
    with (d / "requests.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(SystemExit):
        _run(d, MockClient(), meta)


def test_wrong_model_stops_the_run(tmp_path):
    records, meta, d = _prepared(tmp_path)
    client = MockClient()
    original = client._respond
    client._respond = lambda model, content, finish="stop": original("some-other-model", content, finish)
    with pytest.raises(ModelMismatchError):
        _run(d, client, meta)


# -------------------------------------------------------------------- stats
def test_majority_baseline_macro_f1():
    y = np.array([0] * 620 + [1] * 240)
    assert round(float(S.macro_f1(y, np.zeros_like(y))), 4) == 0.4189


def test_holm_known_values():
    assert np.allclose(S.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_krippendorff_alpha_known_values():
    assert S.krippendorff_alpha_nominal(np.array([[1, 1], [0, 0], [1, 1]])) == pytest.approx(1.0)
    assert S.krippendorff_alpha_nominal(np.array([[1, 1], [0, 0], [1, 0]])) == pytest.approx(4 / 9)


def test_stratified_bootstrap_keeps_class_sizes():
    y = np.array([0] * 70 + [1] * 30)
    idx = S.stratified_indices(y, 50, np.random.default_rng(0))
    assert np.all((y[idx] == 1).sum(axis=1) == 30)


# -------------------------------------------------------------------- probe
def test_probe_eligibility_and_split():
    assert split_post("@USER too short to use") is None
    text = "@USER " + " ".join(f"w{i}" for i in range(12)) + " URL"
    prefix, reference, n_content = split_post(text)
    assert n_content == 12
    assert prefix.split()[-1] == "w5" and reference.split()[0] == "w6"


def test_rouge_l():
    assert rouge_l_f1("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert rouge_l_f1("dog", "the cat sat") == 0.0
    assert 0 < rouge_l_f1("the cat", "the cat sat") < 1


# ----------------------------------------------------------------- analysis
def test_analysis_refuses_incomplete_results(tmp_path):
    records, meta, d = _prepared(tmp_path)
    _run(d, MockClient(truth={r["text"]: r["label"] for r in records}), meta, limit=10)
    with pytest.raises(SystemExit):
        analyse(d, make_figures=False)


def test_baselines_rows():
    from analyze import baselines
    y = np.array([0] * 620 + [1] * 240)
    rows = {r["baseline"].split()[0]: r for r in baselines(y)}
    assert round(rows["majority"]["accuracy"], 4) == 0.7209
    assert round(rows["random,"]["accuracy"], 4) == 0.5976
