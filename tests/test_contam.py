"""Tests for the contamination battery (contam.py). No network, no real posts.

Run with: python -m pytest tests -q
"""

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "promptshift"))

import config as C  # noqa: E402
import contam as K  # noqa: E402
from common import read_jsonl, write_jsonl, sha256_file  # noqa: E402
from mock_api import MockClient, _u  # noqa: E402
from run import run  # noqa: E402


# ------------------------------------------------------------- pure helpers
def test_min_k_prob_takes_lowest_fraction():
    values = [-0.1, -0.2, -5.0, -0.3, -4.0, -0.1, -0.2, -0.1, -0.3, -0.2]
    assert K.min_k_prob(values, k=0.2) == pytest.approx(-4.5)
    assert K.min_k_prob([None, float("nan"), -1.0], k=0.2) == -1.0
    assert K.min_k_prob([], k=0.2) is None


def test_auc_basic_and_ties():
    assert K.auc([3, 4], [1, 2]) == 1.0
    assert K.auc([1, 2], [3, 4]) == 0.0
    assert K.auc([1], [1]) == 0.5


def test_parse_paraphrases_handles_fences_and_garbage():
    assert K.parse_paraphrases('```json\n["a b", "c d", "e f"]\n```') == ["a b", "c d", "e f"]
    assert K.parse_paraphrases("Sure! [\"x\", \"y\", \"z\"] hope this helps") == ["x", "y", "z"]
    assert K.parse_paraphrases("I can't help with that.") is None
    assert K.parse_paraphrases('[1, 2, 3]') is None


def test_paraphrase_validity_rules():
    original = "@USER you are a total clown"
    assert K.valid_paraphrase_set(original, ["@USER you're a complete clown",
                                             "@USER you are an utter clown",
                                             "@USER you are a total joker"])
    assert not K.valid_paraphrase_set(original, ["@USER you are a total clown", "b", "c"])  # copy
    assert not K.valid_paraphrase_set(original, ["a", "a", "c"])                            # duplicate
    assert not K.valid_paraphrase_set(original, ["a", "b"])                                 # too few


def test_parse_letter():
    assert K.parse_letter("B") == "B"
    assert K.parse_letter(" (c) because") == "C"
    assert K.parse_letter("The answer is A") is None
    assert K.parse_letter(None) is None


def test_quiz_position_is_stable_per_item_and_roughly_uniform():
    items = [{"uid": f"g:{i}", "group": "olid_test", "label": 0, "text": f"post number {i} here"}
             for i in range(400)]
    para = [{"row_id": it["uid"], "raw_response": json.dumps([f"p1 {i}", f"p2 {i}", f"p3 {i}"])}
            for i, it in enumerate(items)]
    _, keys1 = K.build_quiz(items, para)
    _, keys2 = K.build_quiz(items[::-1], para)
    assert {k["uid"]: k["answer"] for k in keys1} == {k["uid"]: k["answer"] for k in keys2}
    counts = np.bincount(["ABCD".index(k["answer"]) for k in keys1], minlength=4)
    assert counts.min() > 60


def test_label_probe_prompts_differ_only_in_dataset_identity():
    neutral = K.LABEL_TEMPLATES["neutral"].format(text="x", split="test")
    cue = K.LABEL_TEMPLATES["cue"].format(text="x", split="test")
    assert neutral.split("\n\n", 1)[1] == cue.split("\n\n", 1)[1]
    assert "OLID" in cue and "OLID" not in neutral


def test_logprob_parsers():
    choice = {"logprobs": {"token_logprobs": [None, -1.0, -2.0, -0.5]}}
    assert K.logprobs_from_echo(choice, 3) == [-1.0, -2.0]
    choice = {"prompt_logprobs": [None, {"5": {"logprob": -1.5}}, {"9": {"logprob": -0.5}}]}
    assert K.logprobs_from_prompt_logprobs(choice) == [-1.5, -0.5]


# ------------------------------------------------------------ end-to-end mock
class BatteryMock(MockClient):
    """MockClient plus paraphrase, quiz and completions-endpoint behaviour.

    Planted effects: Llama 'remembers' OLID test posts (label cue helps, quiz above
    chance, higher Min-K%); nothing for fresh posts.
    """

    def __init__(self, members, gold, **kw):
        super().__init__(**kw)
        self.members, self.gold = members, gold
        self.completions = SimpleNamespace(create=self._legacy)

    def _create(self, model, messages, temperature, max_tokens, **kw):
        prompt = messages[-1]["content"]
        if prompt.startswith("Rewrite the following"):
            text = prompt.rsplit('Post: "', 1)[1][:-1]
            return self._respond(model, json.dumps([f"{text} indeed", f"so {text}", f"{text} truly"]))
        if prompt.startswith("The four options"):
            options = {L: prompt.split(f"{L}) ", 1)[1].split("\n", 1)[0] for L in "ABCD"}
            original = [L for L, t in options.items() if t in self.members]
            if model == "llama-3.3-70b" and original and _u(model, prompt) < 0.8:
                return self._respond(model, original[0])
            return self._respond(model, "ABCD"[int(_u(model, prompt, 1) * 4)])
        if "Give the label it has in that dataset" in prompt:
            text = prompt.split('Post: "', 1)[1].split('"\n\n', 1)[0]
            if model == "llama-3.3-70b" and "OLID" in prompt and text in self.members:
                return self._respond(model, "OFFENSIVE" if self.gold[text] else "NOT_OFFENSIVE")
        return super()._create(model, messages, temperature, max_tokens, **kw)

    def _legacy(self, model, prompt, **kw):
        base = -3.0 + (1.0 if (model == "llama-3.3-70b" and prompt in self.members) else 0.0)
        lps = [None] + [base - 2 * _u(model, prompt, i) for i in range(12)]
        if model == "gemma4-26b":
            raise RuntimeError("Out of range float values are not JSON compliant: nan")
        dump = {"model": model, "usage": {"prompt_tokens": 13},
                "choices": [{"logprobs": {"token_logprobs": lps + [-0.1]}}]}
        return SimpleNamespace(model_dump=lambda: dump)


def _posts(prefix, n, seed):
    rng = random.Random(seed)
    words = "you are she is he who what clown liberal gun control maga today really".split()
    return [{"row_id": i, "label": int(rng.random() < 0.3),
             "text": f"@USER {prefix} {i} " + " ".join(rng.choice(words) for _ in range(14))}
            for i in range(n)]


def test_end_to_end_battery_recovers_planted_effects(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "N_BOOTSTRAP", 300)
    monkeypatch.setattr(K, "N_PER_GROUP", 60)
    main_dir = tmp_path / "main"
    monkeypatch.setattr(C, "stage_dir", lambda stage: main_dir)

    groups = {"olid_train": _posts("train", 60, 1), "olid_test": _posts("test", 60, 2),
              "fresh": _posts("fresh", 60, 3)}
    items = []
    for g, recs in groups.items():
        its = K.eligible_items(recs, g)
        write_jsonl(tmp_path / f"items_{g}.jsonl", its)
        items += its
    members = {i["text"] for i in items if i["group"] == "olid_test"}
    gold = {i["text"]: i["label"] for i in items}

    # P1 results for olid_test come from the earlier probe run: simulate an empty one.
    write_jsonl(main_dir / "probe_predictions.jsonl", [])

    client = BatteryMock(members, gold, truth=gold, repeat_noise=0.0)
    manifest = tmp_path / "stage1_requests_all.jsonl"
    write_jsonl(manifest, K.shuffled(K.stage1_requests(items), [1, 2]))
    summary = run(manifest, tmp_path / "stage1_predictions_all.jsonl", tmp_path / "a.jsonl",
                  client, delay=0, backoff_base=0, request_logprobs=False, quiet=True)
    assert summary["remaining"] == 0

    meta = K.quiz_prepare(tmp_path, tmp_path)
    assert meta["n_valid"] == len(items)
    run(tmp_path / "quiz_requests.jsonl", tmp_path / "quiz_predictions.jsonl", tmp_path / "b.jsonl",
        client, delay=0, backoff_base=0, request_logprobs=False, quiet=True)

    n = K.run_minkprob(client, tmp_path, delay=0)
    assert n == 3 * len(items)  # every item attempted for every model; failures recorded, not fatal
    mink = read_jsonl(tmp_path / "mink_results.jsonl")
    assert sum(bool(r.get("failed")) for r in mink if r["model"] == "gemma4-26b") == len(items)

    rows = K.analyse(tmp_path)
    get = {(r["probe"], r["group"], r["model"]): r for r in rows}
    llama = "llama-3.3-70b"
    acc = get[("P3", "olid_test", llama)]
    assert acc["acc_cue"] == pytest.approx(1.0)          # planted: perfect recall with the cue
    assert acc["estimate"] == pytest.approx(1.0 - acc["acc_neutral"])
    assert get[("P3", "fresh", llama)]["estimate"] == pytest.approx(0.0, abs=0.05)
    assert get[("P3", "fresh", "qwen3.6-36b")]["estimate"] == pytest.approx(0.0, abs=0.05)
    assert get[("P5", "olid_test", llama)]["estimate"] > 0.6
    assert get[("P5", "fresh", llama)]["estimate"] < 0.45
    assert get[("P4", "olid_test vs fresh", llama)]["estimate"] > 0.8
    assert 0.3 < get[("P4", "olid_test vs fresh", "qwen3.6-36b")]["estimate"] < 0.7
    summary_md = (tmp_path / "results" / "contam_summary.md").read_text()
    assert "Excluded" in summary_md and "Gemma" in summary_md


def test_mink_exclusion_rule(tmp_path):
    rows = []
    for i in range(100):
        rows.append({"model": "qwen3.6-36b", "uid": f"olid_test:{i}", "group": "olid_test",
                     "min_k_prob": -3.0, **({"failed": True, "min_k_prob": None} if i < 5 else {})})
        rows.append({"model": "gemma4-26b", "uid": f"olid_test:{i}", "group": "olid_test",
                     "min_k_prob": -3.0, **({"failed": True, "min_k_prob": None} if i < 30 else {})})
    write_jsonl(tmp_path / "mink_results.jsonl", rows)
    scores, excluded, fails = K.p4_scores(tmp_path)
    assert excluded == ["gemma4-26b"]            # 30% failed -> excluded
    assert len(scores[("olid_test", "qwen3.6-36b")]) == 95   # 5% failed -> kept
    assert fails["gemma4-26b"] == (30, 100)


def test_parse_letter_long():
    assert K.parse_letter_long("To determine the original, compare... Therefore, the answer is C.") == "C"
    assert K.parse_letter_long("Option B looks natural. **D**") == "D"
    assert K.parse_letter_long("Comparing A and B, I think the original is D") == "D"
    assert K.parse_letter_long("To determine the original") is None
