"""Tests for collect_mastodon.py. No network: the Mastodon client is faked."""

import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "promptshift"))

import collect_mastodon as M  # noqa: E402
from common import read_jsonl  # noqa: E402


def test_status_text_mentions_links_hashtags():
    html = ('<p><span class="h-card"><a href="https://x.social/@bob" class="u-url mention">@<span>bob</span></a></span> '
            'you are wrong, see <a href="https://news.example/a" rel="nofollow"><span class="invisible">https://</span>'
            '<span>news.example/a</span></a> <a href="https://m.s/tags/MAGA" class="mention hashtag">#<span>MAGA</span></a></p>'
            '<p>second &amp; line</p>')
    assert M.status_text(html) == "@USER you are wrong, see URL #MAGA second & line"


def test_stratum_text_first_then_tag():
    assert M.stratum_for("honestly you are a clown", []) == "you are"
    assert M.stratum_for("ban assault weapons now", ["GunControl"]) == "gun control"
    assert M.stratum_for("nothing relevant here", ["news"]) is None
    assert M.stratum_for("the #MAGA crowd again", []) == "MAGA"


def status(text, created="2026-11-10T10:00:00.000Z", lang="en", vis="public", reblog=None, tags=(), sid="1", acct="a"):
    return {"id": sid, "uri": f"https://x/{sid}", "created_at": created, "language": lang, "visibility": vis,
            "reblog": reblog, "content": f"<p>{text}</p>", "tags": [{"name": t} for t in tags],
            "account": {"acct": acct, "url": f"https://x/@{acct}"}}


def test_keep_status_filters():
    assert M.keep_status(status("you are so wrong today"))[0]["keyword"] == "you are"
    assert M.keep_status(status("you are so wrong", created="2026-04-30T23:00:00Z"))[1] == "before_window"
    assert M.keep_status(status("you are so wrong", lang="de"))[1] == "not_english"
    assert M.keep_status(status("you are so wrong", vis="unlisted"))[1] == "not_public"
    assert M.keep_status(status("you are so wrong", reblog={"id": "9"}))[1] == "boost"
    assert M.keep_status(status("just some news today"))[1] == "no_olid_keyword"


class FakeMastodon:
    def __init__(self):
        self.calls = 0

    def tag_page(self, tag, max_id=None):
        self.calls += 1
        page = int(max_id or 0)
        if page >= 6:
            return []
        rng = random.Random(f"{tag}{page}")
        vocab = [f"w{j}" for j in range(500)]
        phrases = ["you are", "she is", "he is", "gun control", "conservatives", "antifa", "MAGA", "liberals"]
        out = []
        for i in range(40):
            month = 11 - (page // 2)
            text = f"{rng.choice(phrases)} " + " ".join(rng.sample(vocab, 8))
            out.append(status(text, created=f"2026-{month:02d}-1{i % 9}T10:00:00Z", sid=f"{tag}{page}{i}",
                              acct=f"{tag}{page}{i}"))
        out[-1]["id"] = str(page + 1)
        return out


def test_collect_resumable_and_sample(tmp_path, monkeypatch):
    raw, progress = tmp_path / "raw_pool.jsonl", tmp_path / "raw_progress.jsonl"
    client = FakeMastodon()
    M.collect(client, delay=0, raw_path=raw, progress_path=progress, salt="s")
    calls = client.calls
    M.collect(client, delay=0, raw_path=raw, progress_path=progress, salt="s")
    assert client.calls == calls
    monkeypatch.setattr(M, "RAW", raw)
    monkeypatch.setattr(M, "PROGRESS", progress)
    meta = M.sample(tmp_path, tmp_path)
    assert meta["n"] == {"calibration": 30, "pilot": 40, "main": 400}
    assert max(meta["sample_by_stratum"].values()) - min(meta["sample_by_stratum"].values()) <= 1
    rows = read_jsonl(tmp_path / "sample_main.jsonl")
    assert all("@" not in r["author_hash"] for r in rows)
