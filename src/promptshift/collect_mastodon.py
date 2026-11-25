"""Backup collector: fresh posts from Mastodon hashtag timelines (no account needed).

Used instead of Bluesky (collect_fresh.py) when Bluesky cannot be reached. Deviation from
PLAN_v3.md 4.2, to be logged: Mastodon offers no full-text search without login, so posts are
gathered from public hashtag timelines and then assigned to OLID's keyword strata by their text.

    python src/promptshift/collect_mastodon.py test      # 1 page per hashtag, counts only
    python src/promptshift/collect_mastodon.py collect   # full pool (resumable per hashtag)
    python src/promptshift/collect_mastodon.py sample    # 470 = 400 main + 40 pilot + 30 calibration

Every post is created on or after 2026-05-01, i.e. after all three models were released.
Post text stays under data/fresh/ (gitignored); authors are stored only as salted hashes.

Code drafted with AI assistance (Claude, Anthropic); the team reviewed it and
is responsible for it (course AI-use policy).
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import collect_fresh as F  # noqa: E402  (shared filters, dedupe, sampling)
from common import append_jsonl, git_state, read_jsonl, sha256_file, sha256_text, utc_now, write_json, write_jsonl  # noqa: E402

INSTANCE = "https://mastodon.social"
# OLID's political keywords as hashtags, plus broad political tags that supply posts for the
# pronoun strata ("you are", "she is", "he is"), which have no hashtag equivalent.
HASHTAGS = ["MAGA", "antifa", "GunControl", "liberals", "conservatives", "politics", "uspolitics", "news"]
TAG_TO_STRATUM = {"maga": "MAGA", "antifa": "antifa", "guncontrol": "gun control",
                  "liberals": "liberals", "conservatives": "conservatives"}
MAX_PAGES = 40            # 40 x 40 = up to 1,600 statuses per hashtag
PAGE_SIZE = 40
REQUEST_DELAY = 1.1       # stays under the anonymous limit of 300 requests / 5 min

RAW = F.FRESH_DIR / "raw_pool.jsonl"
PROGRESS = F.FRESH_DIR / "raw_progress.jsonl"


class _MastodonText(HTMLParser):
    """Status HTML -> plain text: mentions -> @USER, links -> URL, hashtags kept as #tag."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            self.parts.append(" ")
        elif tag == "p":
            self.parts.append(" ")
        elif tag == "a":
            classes = (dict(attrs).get("class") or "").split()
            if "hashtag" in classes:
                kind = "hashtag"
            elif "mention" in classes or "u-url" in classes:
                kind = "mention"
            else:
                kind = "link"
            self.stack.append(kind)
            if kind == "mention":
                self.parts.append("@USER")
            elif kind == "link":
                self.parts.append("URL")

    def handle_endtag(self, tag):
        if tag == "a" and self.stack:
            self.stack.pop()
        elif tag == "p":
            self.parts.append(" ")

    def handle_data(self, data):
        if self.stack and self.stack[-1] in ("mention", "link"):
            return
        self.parts.append(data)


def status_text(html):
    parser = _MastodonText()
    parser.feed(html or "")
    text = unescape("".join(parser.parts))
    text = re.sub(r"https?://\S+|www\.\S+", "URL", text)
    return re.sub(r"\s+", " ", text).strip()


def stratum_for(text, tags):
    """First OLID keyword found in the text; otherwise an OLID keyword hashtag on the status."""
    for keyword in F.KEYWORDS:
        if F.contains_keyword(text, keyword):
            return keyword
    for tag in tags:
        if tag.lower() in TAG_TO_STRATUM:
            return TAG_TO_STRATUM[tag.lower()]
    return None


def keep_status(status):
    """Return (record_without_ids, None) or (None, reason)."""
    if status.get("reblog"):
        return None, "boost"
    if status.get("visibility") != "public":
        return None, "not_public"
    created = status.get("created_at") or ""
    if created < F.EARLIEST:
        return None, "before_window"
    if (status.get("language") or "").lower()[:2] != "en":
        return None, "not_english"
    text = status_text(status.get("content"))
    if len(F.content_words(text)) < F.MIN_CONTENT_WORDS:
        return None, "too_short"
    tags = [t.get("name", "") for t in status.get("tags") or []]
    stratum = stratum_for(text, tags)
    if stratum is None:
        return None, "no_olid_keyword"
    return {"text": text, "keyword": stratum, "created_at": created, "window": created[:7]}, None


class Mastodon:
    def __init__(self, instance=INSTANCE, opener=urlopen):
        self.instance, self.opener = instance, opener

    def tag_page(self, tag, max_id=None):
        params = {"limit": PAGE_SIZE}
        if max_id:
            params["max_id"] = max_id
        url = f"{self.instance}/api/v1/timelines/tag/{tag}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "promptshift/1.0 (university course research)"})
        for attempt in range(1, 6):
            try:
                with self.opener(request, timeout=60) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                if error.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(10 * attempt)
                    continue
                raise SystemExit(f"Mastodon API error {error.code} for #{tag}: "
                                 f"{error.read().decode('utf-8', errors='ignore')[:200]}")
            except URLError as error:
                if attempt < 5:
                    time.sleep(5 * attempt)
                    continue
                raise SystemExit(f"Cannot reach {self.instance}: {error}")
        raise SystemExit("Mastodon API: too many retries")


def collect(client, tags=HASHTAGS, max_pages=MAX_PAGES, delay=REQUEST_DELAY, raw_path=RAW,
            progress_path=PROGRESS, salt=""):
    done = {p["keyword"] for p in read_jsonl(progress_path)}
    seen = {r["uri_hash"] for r in read_jsonl(raw_path)}
    for tag in tags:
        if tag in done:
            continue
        reasons, kept, returned, max_id, oldest = Counter(), 0, 0, None, None
        for _ in range(max_pages):
            statuses = client.tag_page(tag, max_id)
            if not statuses:
                break
            returned += len(statuses)
            for status in statuses:
                uri_hash = sha256_text(status.get("uri") or status.get("url") or str(status.get("id")))
                if uri_hash in seen:
                    reasons["already_in_pool"] += 1
                    continue
                record, reason = keep_status(status)
                if reason:
                    reasons[reason] += 1
                    continue
                seen.add(uri_hash)
                kept += 1
                account = status.get("account") or {}
                append_jsonl(raw_path, {"uri": status.get("uri"), "uri_hash": uri_hash, "source_tag": tag,
                                        "author_hash": F.author_hash(account.get("url") or account.get("acct", ""), salt),
                                        **record})
            max_id = statuses[-1].get("id")
            oldest = statuses[-1].get("created_at")
            if delay:
                time.sleep(delay)
            if oldest and oldest < F.EARLIEST:
                break
        append_jsonl(progress_path, {"keyword": tag, "window": "all", "returned": returned, "kept": kept,
                                     "oldest_seen": oldest, "dropped": dict(reasons), "timestamp_utc": utc_now()})
        print(f"#{tag:14} returned {returned:5d}, kept {kept:5d}, oldest {str(oldest)[:10]}", flush=True)


def sample(fresh_dir=F.FRESH_DIR, protocol_dir=C.PROTOCOL_DIR):
    fresh_dir = Path(fresh_dir)
    if (fresh_dir / "sample_main.jsonl").exists():
        raise SystemExit("sample already drawn; refusing to redraw.")
    pool = read_jsonl(fresh_dir / RAW.name)
    unique = F.dedupe(pool)
    chosen = F.stratified_sample(unique, F.N_MAIN + F.N_PILOT + F.N_CALIB)
    if len(chosen) < F.N_MAIN + F.N_PILOT + F.N_CALIB:
        raise SystemExit(f"only {len(chosen)} usable posts; raise MAX_PAGES or add hashtags.")
    parts = F.split_sample(chosen)
    for name, posts in parts.items():
        write_jsonl(fresh_dir / f"sample_{name}.jsonl", posts)
    progress = read_jsonl(fresh_dir / PROGRESS.name)
    meta = {
        "created_utc": utc_now(), "git": git_state(C.ROOT),
        "source": f"Mastodon hashtag timelines via {INSTANCE} (anonymous public API)",
        "deviation": "Bluesky unreachable; Mastodon has no anonymous full-text search, so posts come from "
                     "hashtag timelines and are assigned to OLID keyword strata by text",
        "hashtags": HASHTAGS, "strata": F.KEYWORDS, "earliest_created": F.EARLIEST, "seed": F.COLLECT_SEED,
        "raw_returned": sum(p["returned"] for p in progress),
        "raw_kept_after_filters": len(pool), "after_dedupe": len(unique),
        "dropped_reasons": dict(sum((Counter(p["dropped"]) for p in progress), Counter())),
        "pool_by_stratum": dict(Counter(p["keyword"] for p in unique)),
        "sample_by_stratum": dict(Counter(p["keyword"] for p in chosen)),
        "sample_by_month": dict(Counter(p["window"] for p in chosen)),
        "sample_by_source_tag": dict(Counter(p.get("source_tag") for p in chosen)),
        "max_posts_per_author": max(Counter(p["author_hash"] for p in chosen).values()),
        "files_sha256": {name: sha256_file(fresh_dir / f"sample_{name}.jsonl") for name in parts},
        "n": {name: len(posts) for name, posts in parts.items()},
    }
    write_json(fresh_dir / "sample_meta.json", meta)
    write_json(Path(protocol_dir) / "fresh_collection_fingerprint.json", meta)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["test", "collect", "sample"])
    args = parser.parse_args()
    if args.step == "test":
        tmp_raw, tmp_progress = F.FRESH_DIR / "test_raw.jsonl", F.FRESH_DIR / "test_progress.jsonl"
        F.FRESH_DIR.mkdir(parents=True, exist_ok=True)
        for path in (tmp_raw, tmp_progress):
            path.unlink(missing_ok=True)
        collect(Mastodon(), max_pages=1, raw_path=tmp_raw, progress_path=tmp_progress, salt="test")
        rows = read_jsonl(tmp_progress)
        print("TEST_DONE", {"returned": sum(r["returned"] for r in rows), "kept": sum(r["kept"] for r in rows),
                            "dropped": dict(sum((Counter(r["dropped"]) for r in rows), Counter())),
                            "kept_by_stratum": dict(Counter(r["keyword"] for r in read_jsonl(tmp_raw)))})
        for path in (tmp_raw, tmp_progress):
            path.unlink(missing_ok=True)
    elif args.step == "collect":
        collect(Mastodon(), salt=F._salt())
        rows = read_jsonl(PROGRESS)
        print("COLLECT_DONE", {"hashtags": len(rows), "kept": sum(r["kept"] for r in rows),
                               "kept_by_stratum": dict(Counter(r["keyword"] for r in read_jsonl(RAW)))})
    else:
        meta = sample()
        print("SAMPLE_OK", {k: meta[k] for k in ("raw_kept_after_filters", "after_dedupe", "n",
                                                  "sample_by_stratum", "sample_by_month", "max_posts_per_author")})


if __name__ == "__main__":
    main()
