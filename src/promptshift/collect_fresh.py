"""Collect fresh posts written after all three models were released (PLAN_v3.md, section 4.2).

Source: Bluesky public posts via app.bsky.feed.searchPosts (needs a free app password).
Window: 2026-05-01 to 2026-11-15. Keywords: OLID's retained keywords that exist on Bluesky.

    read -p "Bluesky handle: " BSKY_HANDLE && export BSKY_HANDLE
    read -s -p "Bluesky app password: " BSKY_APP_PASSWORD && export BSKY_APP_PASSWORD && echo
    python src/promptshift/collect_fresh.py test        # 1 page per keyword, counts only
    python src/promptshift/collect_fresh.py collect     # full pool (resumable)
    python src/promptshift/collect_fresh.py sample      # stratified 470 = 400 main + 40 pilot + 30 calibration

Post text stays under data/fresh/ (gitignored). Author IDs are stored only as salted hashes.
protocol/ gets counts and file hashes only.

Code drafted with AI assistance (Claude, Anthropic); the team reviewed it and
is responsible for it (course AI-use policy).
"""

import argparse
import json
import os
import re
import secrets
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
from common import (  # noqa: E402
    append_jsonl, git_state, read_jsonl, sha256_file, sha256_text, utc_now, write_json,
    write_jsonl,
)

PDS = "https://bsky.social/xrpc"
KEYWORDS = ["you are", "she is", "he is", "gun control", "conservatives", "antifa", "MAGA", "liberals"]
# Dropped from OLID's list (Twitter-only operators): "to:BreitBartNews", "-filter:safe".
WINDOWS = [("2026-05-01", "2026-06-01"), ("2026-06-01", "2026-07-01"), ("2026-07-01", "2026-08-01"),
           ("2026-08-01", "2026-09-01"), ("2026-09-01", "2026-10-01"), ("2026-10-01", "2026-11-01"),
           ("2026-11-01", "2026-11-15")]
EARLIEST = "2026-05-01T00:00:00Z"   # after the April 2026 releases of Gemma 4 and Qwen 3.6
PAGES_PER_CELL = 3                  # up to 300 raw posts per keyword x month
PAGE_SIZE = 100
REQUEST_DELAY = 0.5
COLLECT_SEED = 20261129
N_MAIN, N_PILOT, N_CALIB = 400, 40, 30
MAX_PER_AUTHOR = 2
MIN_CONTENT_WORDS = 3
NEAR_DUP_JACCARD = 0.8

FRESH_DIR = C.DATA_DIR / "fresh"
RAW = FRESH_DIR / "raw_pool.jsonl"          # contains URIs and text: gitignored, delete after study
PROGRESS = FRESH_DIR / "raw_progress.jsonl"
SALT = FRESH_DIR / "salt.txt"


# ------------------------------------------------------------------ text processing
def normalise_post(text, facets=None):
    """OLID-style: mentions -> @USER, links -> URL, whitespace collapsed.

    Bluesky shortens link text (e.g. 'nytimes.com/2026/...'), so link and mention spans
    are taken from the record's facets (UTF-8 byte offsets); regexes catch the rest."""
    data = text.encode("utf-8")
    spans = []
    for facet in facets or []:
        index = facet.get("index") or {}
        start, end = index.get("byteStart"), index.get("byteEnd")
        if start is None or end is None or not (0 <= start < end <= len(data)):
            continue
        kinds = {f.get("$type", "") for f in facet.get("features") or []}
        if any(k.endswith("#link") for k in kinds):
            spans.append((start, end, "URL"))
        elif any(k.endswith("#mention") for k in kinds):
            spans.append((start, end, "@USER"))
    out, cursor = b"", 0
    for start, end, token in sorted(spans):
        if start < cursor:
            continue
        out += data[cursor:start] + token.encode()
        cursor = end
    out += data[cursor:]
    result = out.decode("utf-8", errors="ignore")
    result = re.sub(r"https?://\S+|www\.\S+", "URL", result)
    result = re.sub(r"(?<![\w@])@[A-Za-z0-9][\w.-]*\.[A-Za-z]{2,}", "@USER", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def content_words(text):
    return [w for w in re.findall(r"[\w']+", text) if w not in {"USER", "URL"}]


def contains_keyword(text, keyword):
    return re.search(r"(?<![\w])" + re.escape(keyword) + r"(?![\w])", text, flags=re.IGNORECASE) is not None


def token_set(text):
    return set(w.lower() for w in content_words(text))


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


def author_hash(did, salt):
    return sha256_text(salt + did)[:16]


def keep_post(post, keyword, window):
    """Return the normalised record, or None with a reason."""
    record = post.get("record") or {}
    created = record.get("createdAt") or ""
    if not created or created < EARLIEST or not (window[0] <= created[:10] < window[1]):
        return None, "outside_window"
    langs = record.get("langs") or []
    if "en" not in [str(l).lower()[:2] for l in langs]:
        return None, "not_english"
    embed_type = (record.get("embed") or {}).get("$type", "")
    if "embed.record" in embed_type:
        return None, "quote_post"
    text = normalise_post(record.get("text") or "", record.get("facets"))
    if len(content_words(text)) < MIN_CONTENT_WORDS:
        return None, "too_short"
    if not contains_keyword(text, keyword):
        return None, "keyword_not_in_text"
    return text, None


# ------------------------------------------------------------------ API
class Bluesky:
    def __init__(self, handle, password, opener=urlopen):
        self.handle, self.password, self.opener = handle, password, opener
        self.token = None

    def _json(self, request):
        with self.opener(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def login(self):
        body = json.dumps({"identifier": self.handle, "password": self.password}).encode()
        request = Request(f"{PDS}/com.atproto.server.createSession", data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
        self.token = self._json(request)["accessJwt"]

    def search(self, q, since, until, cursor=None):
        params = {"q": q, "sort": "latest", "lang": "en", "limit": PAGE_SIZE,
                  "since": f"{since}T00:00:00Z", "until": f"{until}T00:00:00Z"}
        if cursor:
            params["cursor"] = cursor
        for attempt in range(1, 6):
            if self.token is None:
                self.login()
            request = Request(f"{PDS}/app.bsky.feed.searchPosts?{urlencode(params)}",
                              headers={"Authorization": f"Bearer {self.token}"})
            try:
                return self._json(request)
            except HTTPError as error:
                body = error.read().decode("utf-8", errors="ignore")[:300]
                if error.code in (400, 401) and "xpired" in body:
                    self.token = None
                    continue
                if error.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise SystemExit(f"Bluesky API error {error.code}: {body}")
        raise SystemExit("Bluesky API: too many retries")


def query_for(keyword):
    return f'"{keyword}"' if " " in keyword else keyword


def collect(client, pages=PAGES_PER_CELL, windows=WINDOWS, keywords=KEYWORDS, delay=REQUEST_DELAY,
            raw_path=RAW, progress_path=PROGRESS, salt=None):
    """Fetch every keyword x month cell (resumable by cell)."""
    done = {(p["keyword"], p["window"]) for p in read_jsonl(progress_path)}
    seen = {r["uri_hash"] for r in read_jsonl(raw_path)}
    for keyword in keywords:
        for window in windows:
            cell = (keyword, window[0])
            if cell in done:
                continue
            reasons, kept, returned, cursor = Counter(), 0, 0, None
            for _ in range(pages):
                page = client.search(query_for(keyword), window[0], window[1], cursor)
                posts = page.get("posts") or []
                returned += len(posts)
                for post in posts:
                    uri_hash = sha256_text(post.get("uri", ""))
                    if uri_hash in seen:
                        reasons["already_in_pool"] += 1
                        continue
                    text, reason = keep_post(post, keyword, window)
                    if reason:
                        reasons[reason] += 1
                        continue
                    seen.add(uri_hash)
                    kept += 1
                    append_jsonl(raw_path, {
                        "uri": post.get("uri"), "uri_hash": uri_hash, "keyword": keyword,
                        "window": window[0], "created_at": post["record"]["createdAt"],
                        "author_hash": author_hash((post.get("author") or {}).get("did", ""), salt),
                        "text": text})
                cursor = page.get("cursor")
                if delay:
                    time.sleep(delay)
                if not cursor or not posts:
                    break
            append_jsonl(progress_path, {"keyword": keyword, "window": window[0], "returned": returned,
                                         "kept": kept, "dropped": dict(reasons), "timestamp_utc": utc_now()})
            print(f"{keyword!r:16} {window[0]}: returned {returned:4d}, kept {kept:4d}", flush=True)


# ------------------------------------------------------------------ sampling
def dedupe(pool):
    """Exact duplicates (normalised, case-insensitive) and near-duplicates (Jaccard >= 0.8)."""
    kept, sets, exact = [], [], set()
    for post in sorted(pool, key=lambda p: p["created_at"]):
        key = " ".join(content_words(post["text"])).lower()
        if key in exact:
            continue
        s = token_set(post["text"])
        if any(jaccard(s, t) >= NEAR_DUP_JACCARD for t in sets):
            continue
        exact.add(key)
        sets.append(s)
        kept.append(post)
    return kept


def stratified_sample(pool, n_total, seed=COLLECT_SEED, keywords=KEYWORDS):
    """Equal share per keyword, spread across months, at most MAX_PER_AUTHOR per author."""
    rng = np.random.default_rng(seed)
    by_cell = defaultdict(list)
    for post in pool:
        by_cell[(post["keyword"], post["window"])].append(post)
    for cell in by_cell:
        order = rng.permutation(len(by_cell[cell]))
        by_cell[cell] = [by_cell[cell][i] for i in order]
    quota = {k: n_total // len(keywords) + (1 if i < n_total % len(keywords) else 0)
             for i, k in enumerate(keywords)}
    authors, chosen = Counter(), []
    shortfall = 0
    for keyword in keywords:
        cells = [c for c in sorted(by_cell) if c[0] == keyword]
        picked = 0
        while picked < quota[keyword] and any(by_cell[c] for c in cells):
            for cell in cells:
                if picked >= quota[keyword]:
                    break
                while by_cell[cell]:
                    post = by_cell[cell].pop()
                    if authors[post["author_hash"]] < MAX_PER_AUTHOR:
                        authors[post["author_hash"]] += 1
                        chosen.append(post)
                        picked += 1
                        break
        shortfall += quota[keyword] - picked
    if shortfall:  # refill from any keyword, still respecting the author cap
        rest = [p for cell in sorted(by_cell) for p in by_cell[cell]]
        for i in rng.permutation(len(rest)):
            if shortfall == 0:
                break
            post = rest[i]
            if authors[post["author_hash"]] < MAX_PER_AUTHOR:
                authors[post["author_hash"]] += 1
                chosen.append(post)
                shortfall -= 1
    return chosen


def split_sample(chosen, seed=COLLECT_SEED):
    rng = np.random.default_rng([seed, 1])
    order = [chosen[i] for i in rng.permutation(len(chosen))]
    parts = {"calibration": order[:N_CALIB], "pilot": order[N_CALIB:N_CALIB + N_PILOT],
             "main": order[N_CALIB + N_PILOT:N_CALIB + N_PILOT + N_MAIN]}
    out = {}
    for name, posts in parts.items():
        out[name] = [{"row_id": f"{name}{i:04d}", "text": p["text"], "keyword": p["keyword"],
                      "window": p["window"], "created_at": p["created_at"],
                      "author_hash": p["author_hash"]} for i, p in enumerate(posts)]
    return out


def sample(fresh_dir=FRESH_DIR, protocol_dir=C.PROTOCOL_DIR):
    fresh_dir = Path(fresh_dir)
    if (fresh_dir / "sample_main.jsonl").exists():
        raise SystemExit("sample already drawn; refusing to redraw (delete data/fresh/sample_*.jsonl only "
                         "if no labelling has started).")
    pool = read_jsonl(fresh_dir / RAW.name)
    unique = dedupe(pool)
    chosen = stratified_sample(unique, N_MAIN + N_PILOT + N_CALIB)
    if len(chosen) < N_MAIN + N_PILOT + N_CALIB:
        raise SystemExit(f"only {len(chosen)} usable posts; collect more before sampling.")
    parts = split_sample(chosen)
    for name, posts in parts.items():
        write_jsonl(fresh_dir / f"sample_{name}.jsonl", posts)
    progress = read_jsonl(fresh_dir / PROGRESS.name)
    meta = {
        "created_utc": utc_now(), "git": git_state(C.ROOT), "source": "Bluesky app.bsky.feed.searchPosts",
        "window": [WINDOWS[0][0], WINDOWS[-1][1]], "keywords": KEYWORDS, "seed": COLLECT_SEED,
        "raw_returned": sum(p["returned"] for p in progress),
        "raw_kept_after_filters": len(pool), "after_dedupe": len(unique),
        "dropped_reasons": dict(sum((Counter(p["dropped"]) for p in progress), Counter())),
        "pool_by_keyword": dict(Counter(p["keyword"] for p in unique)),
        "sample_by_keyword": dict(Counter(p["keyword"] for p in chosen)),
        "sample_by_month": dict(Counter(p["window"] for p in chosen)),
        "max_posts_per_author": max(Counter(p["author_hash"] for p in chosen).values()),
        "files_sha256": {name: sha256_file(fresh_dir / f"sample_{name}.jsonl") for name in parts},
        "n": {name: len(posts) for name, posts in parts.items()},
    }
    write_json(fresh_dir / "sample_meta.json", meta)
    write_json(Path(protocol_dir) / "fresh_collection_fingerprint.json", meta)
    return meta


# ------------------------------------------------------------------ main
def _client():
    handle, password = os.environ.get("BSKY_HANDLE"), os.environ.get("BSKY_APP_PASSWORD")
    if not handle or not password:
        raise SystemExit("Set BSKY_HANDLE and BSKY_APP_PASSWORD first (see the top of this file).")
    return Bluesky(handle, password)


def _salt():
    FRESH_DIR.mkdir(parents=True, exist_ok=True)
    if not SALT.exists():
        SALT.write_text(secrets.token_hex(16), encoding="utf-8")
    return SALT.read_text(encoding="utf-8").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["test", "collect", "sample"])
    args = parser.parse_args()
    if args.step == "test":
        client = _client()
        client.login()
        print("LOGIN_OK")
        tmp_raw, tmp_progress = FRESH_DIR / "test_raw.jsonl", FRESH_DIR / "test_progress.jsonl"
        for path in (tmp_raw, tmp_progress):
            path.unlink(missing_ok=True)
        collect(client, pages=1, windows=WINDOWS[:1], raw_path=tmp_raw, progress_path=tmp_progress,
                salt=_salt())
        rows = read_jsonl(tmp_progress)
        print("TEST_DONE", {"returned": sum(r["returned"] for r in rows), "kept": sum(r["kept"] for r in rows),
                            "dropped": dict(sum((Counter(r["dropped"]) for r in rows), Counter()))})
        for path in (tmp_raw, tmp_progress):
            path.unlink(missing_ok=True)
    elif args.step == "collect":
        collect(_client(), salt=_salt())
        rows = read_jsonl(PROGRESS)
        print("COLLECT_DONE", {"cells": len(rows), "kept": sum(r["kept"] for r in rows)})
    else:
        meta = sample()
        print("SAMPLE_OK", {k: meta[k] for k in ("raw_kept_after_filters", "after_dedupe", "n",
                                                  "sample_by_keyword", "max_posts_per_author")})


if __name__ == "__main__":
    main()
