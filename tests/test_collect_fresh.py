"""Tests for collect_fresh.py. No network: the Bluesky client is faked."""

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "promptshift"))

import collect_fresh as F  # noqa: E402
from common import read_jsonl  # noqa: E402


def facet(text, sub, kind):
    start = text.encode().index(sub.encode())
    return {"index": {"byteStart": start, "byteEnd": start + len(sub.encode())},
            "features": [{"$type": f"app.bsky.richtext.facet#{kind}"}]}


def test_normalise_uses_facets_with_multibyte_text():
    text = "🔥 @alice.bsky.social you are wrong, read nytimes.com/2026/05/x... now"
    facets = [facet(text, "@alice.bsky.social", "mention"), facet(text, "nytimes.com/2026/05/x...", "link")]
    assert F.normalise_post(text, facets) == "🔥 @USER you are wrong, read URL now"


def test_normalise_regex_fallback_and_whitespace():
    assert F.normalise_post("hi @bob.bsky.social\n\nsee https://x.org/a  ok") == "hi @USER see URL ok"


def test_keyword_word_boundaries():
    assert F.contains_keyword("Honestly YOU ARE great", "you are")
    assert not F.contains_keyword("you aren't great", "you are")
    assert F.contains_keyword("those MAGA folks", "MAGA")
    assert not F.contains_keyword("magazine", "MAGA")


def post(text, created="2026-05-10T10:00:00Z", langs=("en",), embed=None, did="did:plc:a", uri="at://1"):
    record = {"text": text, "createdAt": created, "langs": list(langs)}
    if embed:
        record["embed"] = {"$type": embed}
    return {"uri": uri, "author": {"did": did}, "record": record}


def test_keep_post_filters():
    w = ("2026-05-01", "2026-06-01")
    assert F.keep_post(post("you are so wrong today"), "you are", w)[0] == "you are so wrong today"
    assert F.keep_post(post("you are wrong", created="2026-04-30T23:59:00Z"), "you are", w)[1] == "outside_window"
    assert F.keep_post(post("you are wrong", langs=("de",)), "you are", w)[1] == "not_english"
    assert F.keep_post(post("you are wrong", embed="app.bsky.embed.record"), "you are", w)[1] == "quote_post"
    assert F.keep_post(post("you are"), "you are", w)[1] == "too_short"
    assert F.keep_post(post("gun laws are bad here"), "gun control", w)[1] == "keyword_not_in_text"


def test_dedupe_exact_and_near():
    pool = [{"text": "you are a clown and a fool today", "created_at": "1"},
            {"text": "You are a clown and a fool today!", "created_at": "2"},
            {"text": "you are a clown and a fool today ok", "created_at": "3"},
            {"text": "she is completely different here", "created_at": "4"}]
    assert [p["created_at"] for p in F.dedupe(pool)] == ["1", "4"]


def test_stratified_sample_quotas_and_author_cap():
    pool = []
    for k in F.KEYWORDS:
        for w, _ in F.WINDOWS:
            for i in range(10):
                pool.append({"keyword": k, "window": w, "author_hash": f"{k}{i % 4}",
                             "text": f"{k} {w} {i}", "created_at": w})
    chosen = F.stratified_sample(pool, 40)
    assert len(chosen) == 40
    assert set(Counter(p["keyword"] for p in chosen).values()) == {5}
    assert max(Counter(p["author_hash"] for p in chosen).values()) <= F.MAX_PER_AUTHOR
    assert len(set(p["window"] for p in chosen)) >= 4


class FakeClient:
    def __init__(self):
        self.calls = 0

    def search(self, q, since, until, cursor=None):
        self.calls += 1
        kw = q.strip('"')
        page = int(cursor or 0)
        import random
        rng = random.Random(f"{kw}{since}{page}")
        vocab = [f"w{j}" for j in range(400)]
        posts = [post(f"{kw} " + " ".join(rng.sample(vocab, 8)), created=f"{since}T12:00:00Z",
                      did=f"did:{kw}:{since}:{i}", uri=f"at://{kw}/{since}/{page}/{i}") for i in range(20)]
        return {"posts": posts, "cursor": str(page + 1) if page < 1 else None}


def test_collect_is_resumable_and_sample_end_to_end(tmp_path, monkeypatch):
    raw, progress = tmp_path / "raw_pool.jsonl", tmp_path / "raw_progress.jsonl"
    client = FakeClient()
    F.collect(client, delay=0, raw_path=raw, progress_path=progress, salt="s")
    calls = client.calls
    F.collect(client, delay=0, raw_path=raw, progress_path=progress, salt="s")
    assert client.calls == calls  # every cell already done
    assert all("did:" not in r["author_hash"] for r in read_jsonl(raw))
    monkeypatch.setattr(F, "RAW", raw)
    monkeypatch.setattr(F, "PROGRESS", progress)
    meta = F.sample(tmp_path, tmp_path)
    assert meta["n"] == {"calibration": 30, "pilot": 40, "main": 400}
    ids = [r["row_id"] for n in ("calibration", "pilot", "main") for r in read_jsonl(tmp_path / f"sample_{n}.jsonl")]
    assert len(ids) == len(set(ids)) == 470
    with pytest.raises(SystemExit):
        F.sample(tmp_path, tmp_path)  # refuses to redraw
