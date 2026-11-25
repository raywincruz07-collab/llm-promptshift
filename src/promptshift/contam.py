"""Contamination evidence battery (PLAN_v3.md, section 7).

Groups
  olid_train  200 random eligible posts from the TweetEval/OLID TRAIN split
  olid_test   the same 200 TEST posts used by probe.py (P1 results reused)
  fresh       posts written after the models' release (added later, with gold labels)

Probes
  P1  guided vs general completion (Golchin & Surdeanu 2024)        [train, fresh; test reused]
  P3  dataset-cue label probe: "label it has in OLID" vs matched
      "label it has in that dataset" prompt                           [all groups]
  P4  Min-K% Prob (Shi et al. 2024), k = 20%, from input-token
      log probabilities via the completions endpoint                  [all groups]
  P5  Data Contamination Quiz (Golchin & Surdeanu): pick the original
      among 3 paraphrases written by an independent model             [all groups]

Commands (in order)
  python src/promptshift/contam.py prepare --groups olid_train olid_test
  python src/promptshift/contam.py run --part stage1 --all
  python src/promptshift/contam.py quiz-prepare
  python src/promptshift/contam.py run --part quiz --all
  python src/promptshift/contam.py quiz-long-prepare      # Qwen only, see quiz_long_prepare
  python src/promptshift/contam.py run --part quiz_long --all
  python src/promptshift/contam.py minkprob --all
  python src/promptshift/contam.py analyze

All interpretation rules are fixed in PLAN_v3.md before the results exist.
A probe that finds nothing means "no evidence detected", never "not in the training data".

Code drafted with AI assistance (Claude, Anthropic); the team reviewed it and
is responsible for it (course AI-use policy).
"""

import argparse
import csv
import json
import math
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
from common import (  # noqa: E402
    append_jsonl, git_state, read_json, read_jsonl, sha256_file, sha256_text,
    utc_now, write_json, write_jsonl,
)
from probe import (  # noqa: E402
    GENERAL_TEMPLATE, GUIDED_TEMPLATE, LABEL_NAME, clean_completion, normalise,
    rouge_l_f1, split_post,
)

# ------------------------------------------------------------------ settings
CONTAM_SEED = 20261128
N_PER_GROUP = 200
MINK_K = 0.20
PARAPHRASE_MODEL = "ministral-3-14b"   # different family from all three tested models
MAX_TOKENS_LABEL = C.MAX_TOKENS_CLASSIFY
MAX_TOKENS_PARAPHRASE = 600
MAX_TOKENS_QUIZ = 4
MINK_MODELS = C.MODELS               # Gemma is attempted; excluded if the API cannot serve it
GROUPS = ["olid_train", "olid_test", "fresh"]
GROUP_SPLIT_NAME = {"olid_train": "train", "olid_test": "test", "fresh": "test"}
# "fresh" deliberately receives the same claim ("test split of OLID") so that any
# effect of the cue itself, without memorisation, shows up in the control group.

CONTAM_DIR = C.DATA_DIR / "contam"
FRESH_POSTS = C.DATA_DIR / "fresh" / "gold_main.jsonl"   # created in phase D

LABEL_TEMPLATES = {
    "neutral": (
        "The following social-media post comes from a dataset of social-media posts. "
        "Give the label it has in that dataset.\n\n"
        'Post: "{text}"\n\n'
        "Answer with exactly one label:\nOFFENSIVE or NOT_OFFENSIVE."
    ),
    "cue": (
        "The following social-media post comes from the {split} split of the TweetEval "
        "offensive-language dataset (OLID). Give the label it has in that dataset.\n\n"
        'Post: "{text}"\n\n'
        "Answer with exactly one label:\nOFFENSIVE or NOT_OFFENSIVE."
    ),
}

PARAPHRASE_TEMPLATE = (
    "Rewrite the following social-media post in three different ways. In each version, "
    "replace several words with synonyms or near-synonyms, but keep the meaning, the order "
    "of ideas, the tone, any offensive words, and the tokens @USER and URL unchanged. Each "
    "version must differ from the original and from the other versions.\n"
    "Return only a JSON list of three strings, nothing else.\n\n"
    'Post: "{text}"'
)

QUIZ_TEMPLATE = (
    "The four options below are versions of one social-media post. Exactly one option is "
    "the original post as it appears in the {split} split of the TweetEval "
    "offensive-language dataset (OLID); the other three are paraphrases. "
    "Which option is the original?\n\n"
    "A) {A}\nB) {B}\nC) {C}\nD) {D}\n\n"
    "Answer with exactly one letter: A, B, C or D."
)
LETTERS = "ABCD"


# ------------------------------------------------------------------- items
def download_train_split(cache_dir=C.CACHE_DIR):
    """The train split is not pinned in config; its hash is recorded instead."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "train-00000-of-00001.parquet"
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
    return path


def eligible_items(records, group):
    items = []
    for record in records:
        split = split_post(record["text"])
        if split:
            prefix, reference, n_content = split
            items.append({"group": group, "row_id": record["row_id"],
                          "uid": f"{group}:{record['row_id']}", "label": int(record["label"]),
                          "text": record["text"], "prefix": prefix, "reference": reference,
                          "n_content_tokens": n_content})
    return items


def sample_items(items, n, seed):
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(items), size=min(n, len(items)), replace=False))
    return [items[i] for i in chosen]


def build_group(group):
    if group == "olid_train":
        import pandas as pd
        path = download_train_split()
        frame = pd.read_parquet(path)
        records = [{"row_id": i, "label": int(r.label), "text": r.text}
                   for i, r in enumerate(frame.itertuples(index=False))]
        source = {"file_sha256": sha256_file(path), "n_rows": len(records),
                  "label_counts": dict(Counter(r["label"] for r in records))}
        items = sample_items(eligible_items(records, group), N_PER_GROUP, [CONTAM_SEED, 1])
        return items, source
    if group == "olid_test":
        main_dir = C.stage_dir("main")
        sample = {r["row_id"]: r for r in read_jsonl(main_dir / "sample.jsonl")}
        chosen = [i["row_id"] for i in read_jsonl(main_dir / "probe_items.jsonl")]
        if not chosen:
            raise SystemExit("data/main/probe_items.jsonl missing; olid_test reuses the P1 sample.")
        items = eligible_items([sample[r] for r in chosen], group)
        return items, {"reused_from": "data/main/probe_items.jsonl", "n": len(items)}
    if group == "fresh":
        records = read_jsonl(FRESH_POSTS)
        if not records:
            raise SystemExit(f"{FRESH_POSTS} not found: fresh posts need gold labels first (phase D).")
        items = eligible_items(records, group)
        return items, {"file_sha256": sha256_file(FRESH_POSTS), "n_rows": len(records),
                       "n_eligible": len(items)}
    raise ValueError(group)


def make_request(rid, task, model, uid, condition, max_tokens, prompt, sample):
    return {"request_id": rid, "task": task, "sample": sample, "model": model,
            "row_id": uid, "condition": condition, "max_tokens": max_tokens,
            "prompt_text": prompt, "prompt_sha256": sha256_text(prompt)}


def stage1_requests(items):
    """P1 (not for olid_test, reused), P3 label probe, paraphrase generation."""
    requests = []
    for item in items:
        group, uid = item["group"], item["uid"]
        split = GROUP_SPLIT_NAME[group]
        for model in C.MODELS:
            if group != "olid_test":
                for kind, template in (("general", GENERAL_TEMPLATE), ("guided", GUIDED_TEMPLATE)):
                    prompt = template.format(split=split, label=LABEL_NAME[item["label"]],
                                             prefix=item["prefix"])
                    requests.append(make_request(f"p1:{model}:{uid}:{kind}", "complete", model,
                                                 uid, kind, C.MAX_TOKENS_PROBE, prompt, group))
            for kind, template in LABEL_TEMPLATES.items():
                prompt = template.format(split=split, text=item["text"])
                requests.append(make_request(f"p3:{model}:{uid}:{kind}", "classify", model,
                                             uid, kind, MAX_TOKENS_LABEL, prompt, group))
        prompt = PARAPHRASE_TEMPLATE.format(text=item["text"])
        requests.append(make_request(f"para:{PARAPHRASE_MODEL}:{uid}", "paraphrase",
                                     PARAPHRASE_MODEL, uid, "paraphrase",
                                     MAX_TOKENS_PARAPHRASE, prompt, group))
    return requests


def shuffled(requests, seed):
    order = np.random.default_rng(seed).permutation(len(requests))
    out = [requests[i] for i in order]
    for index, request in enumerate(out):
        request["order"] = index
    return out


def prepare(groups, out_dir=CONTAM_DIR, protocol_dir=C.PROTOCOL_DIR):
    out_dir = Path(out_dir)
    tag = "_".join(groups)
    if read_jsonl(out_dir / f"stage1_predictions_{tag}.jsonl"):
        raise SystemExit("stage1 already has results for these groups; refusing to rebuild.")
    already = [g for g in groups if (out_dir / f"items_{g}.jsonl").exists()
               and any(g in t for t in _tags_with_results(out_dir))]
    if already:
        raise SystemExit(f"groups {already} already have results in another manifest.")
    all_items, sources = [], {}
    for group in groups:
        items, source = build_group(group)
        write_jsonl(out_dir / f"items_{group}.jsonl", items)
        sources[group] = source | {"n_items": len(items),
                                   "uids_sha256": sha256_text(",".join(i["uid"] for i in items))}
        all_items += items
    requests = shuffled(stage1_requests(all_items), [CONTAM_SEED, len(tag)])
    manifest = out_dir / f"stage1_requests_{tag}.jsonl"
    write_jsonl(manifest, requests)
    meta = {"created_utc": utc_now(), "git": git_state(C.ROOT), "groups": groups,
            "sources": sources, "n_requests": len(requests),
            "requests_file": manifest.name, "requests_file_sha256": sha256_file(manifest),
            "templates_sha256": {k: sha256_text(v) for k, v in
                                 {**LABEL_TEMPLATES, "paraphrase": PARAPHRASE_TEMPLATE,
                                  "quiz": QUIZ_TEMPLATE}.items()},
            "settings": {"seed": CONTAM_SEED, "n_per_group": N_PER_GROUP, "mink_k": MINK_K,
                         "paraphrase_model": PARAPHRASE_MODEL}}
    write_json(out_dir / f"stage1_meta_{'_'.join(groups)}.json", meta)
    write_json(protocol_dir / f"contam_stage1_fingerprint_{'_'.join(groups)}.json",
               {k: v for k, v in meta.items()})
    return meta


def _tags_with_results(out_dir):
    return [p.stem.replace("stage1_predictions_", "") for p in Path(out_dir).glob("stage1_predictions_*.jsonl")
            if read_jsonl(p)]


def stage1_all(out_dir):
    records = []
    for path in sorted(Path(out_dir).glob("stage1_predictions_*.jsonl")):
        records += read_jsonl(path)
    return records


# ------------------------------------------------------------------- quiz
def parse_paraphrases(text):
    """Return a list of paraphrase strings, or None if the reply is unusable."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        return None
    return [v.strip() for v in value]


def valid_paraphrase_set(original, paraphrases):
    """Exactly 3, all different from the original and from each other, none empty."""
    if not paraphrases or len(paraphrases) != 3:
        return False
    keys = [" ".join(normalise(p)) for p in paraphrases]
    original_key = " ".join(normalise(original))
    if any(not k for k in keys) or original_key in keys or len(set(keys)) != 3:
        return False
    return True


def build_quiz(items, paraphrase_records, seed=CONTAM_SEED):
    """Return (requests, key rows). Original placed at a seeded random position."""
    by_uid = {r["row_id"]: r for r in paraphrase_records}
    requests, keys = [], []
    for item in items:
        record = by_uid.get(item["uid"])
        paraphrases = parse_paraphrases(record["raw_response"]) if record else None
        ok = valid_paraphrase_set(item["text"], paraphrases)
        keys.append({"uid": item["uid"], "group": item["group"], "label": item["label"],
                     "quiz_valid": ok})
        if not ok:
            continue
        position = int(np.random.default_rng([seed, 7, int(sha256_text(item["uid"])[:8], 16)]).integers(0, 4))
        options = list(paraphrases)
        options.insert(position, item["text"])
        keys[-1]["answer"] = LETTERS[position]
        prompt = QUIZ_TEMPLATE.format(split=GROUP_SPLIT_NAME[item["group"]],
                                      **dict(zip(LETTERS, options)))
        for model in C.MODELS:
            requests.append(make_request(f"p5:{model}:{item['uid']}", "quiz", model,
                                         item["uid"], "quiz", MAX_TOKENS_QUIZ, prompt,
                                         item["group"]))
    return requests, keys


def parse_letter(text):
    if not isinstance(text, str):
        return None
    match = re.match(r"\s*\(?([ABCD])\b", text.strip().upper())
    return match.group(1) if match else None


QUIZ_LONG_MODELS = ["qwen3.6-36b"]
MAX_TOKENS_QUIZ_LONG = 256


def parse_letter_long(text):
    """For replies that reason before answering: explicit 'answer is X' first,
    then a bold/parenthesised letter, then the last standalone letter."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    for pattern in (r"answer\s*(?:is|:)?\s*[:\-]?\s*\**\s*(?:option\s*)?\(?([ABCD])\b",
                    r"\*\*\s*\(?([ABCD])\)?\s*\*\*",
                    r"\boption\s*\(?([ABCD])\)?\s*is\s+the\s+original"):
        found = re.findall(pattern, t, flags=re.IGNORECASE)
        if found:
            return found[-1].upper()
    found = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", t)
    return found[-1] if found else None


def quiz_long_prepare(out_dir=CONTAM_DIR, protocol_dir=C.PROTOCOL_DIR):
    """Deviation (20 Nov): Qwen began reasoning and was cut off at 4 tokens in 105/391
    quiz answers. Re-ask Qwen the identical quiz prompts with room to answer."""
    out_dir = Path(out_dir)
    base = read_jsonl(out_dir / "quiz_requests.jsonl")
    requests = []
    for r in base:
        if r["model"] in QUIZ_LONG_MODELS:
            requests.append({**r, "request_id": r["request_id"].replace("p5:", "p5long:"),
                             "condition": "quiz_long", "max_tokens": MAX_TOKENS_QUIZ_LONG})
    requests = shuffled(requests, [CONTAM_SEED, 9])
    write_jsonl(out_dir / "quiz_long_requests.jsonl", requests)
    meta = {"created_utc": utc_now(), "models": QUIZ_LONG_MODELS, "max_tokens": MAX_TOKENS_QUIZ_LONG,
            "n_requests": len(requests),
            "requests_file_sha256": sha256_file(out_dir / "quiz_long_requests.jsonl"),
            "reason": "Qwen replies truncated at 4 tokens before giving a letter"}
    write_json(out_dir / "quiz_long_meta.json", meta)
    write_json(Path(protocol_dir) / "contam_quiz_long_fingerprint.json", meta)
    return meta


def quiz_prepare(out_dir=CONTAM_DIR, protocol_dir=C.PROTOCOL_DIR):
    out_dir = Path(out_dir)
    items = [i for g in GROUPS for i in read_jsonl(out_dir / f"items_{g}.jsonl")]
    para = [r for r in stage1_all(out_dir) if r["task"] == "paraphrase"]
    done = {r["row_id"] for r in para}
    items = [i for i in items if i["uid"] in done]
    if not items:
        raise SystemExit("no paraphrases yet: run stage1 first.")
    requests, keys = build_quiz(items, para)
    if read_jsonl(out_dir / "quiz_predictions.jsonl"):
        old = {r["request_id"] for r in read_jsonl(out_dir / "quiz_requests.jsonl")}
        new = [r for r in requests if r["request_id"] not in old]
        requests = read_jsonl(out_dir / "quiz_requests.jsonl") + new
        if not new:
            print("quiz already prepared; nothing new.")
    requests = shuffled(requests, [CONTAM_SEED, 8])
    write_jsonl(out_dir / "quiz_requests.jsonl", requests)
    write_jsonl(out_dir / "quiz_key.jsonl", keys)
    meta = {"created_utc": utc_now(), "n_items": len(keys),
            "n_valid": sum(k["quiz_valid"] for k in keys),
            "valid_by_group": {g: [sum(k["quiz_valid"] for k in keys if k["group"] == g),
                                   sum(1 for k in keys if k["group"] == g)] for g in GROUPS},
            "n_requests": len(requests),
            "requests_file_sha256": sha256_file(out_dir / "quiz_requests.jsonl"),
            "key_file_sha256": sha256_file(out_dir / "quiz_key.jsonl")}
    write_json(out_dir / "quiz_meta.json", meta)
    write_json(Path(protocol_dir) / "contam_quiz_fingerprint.json", meta)
    return meta


# --------------------------------------------------------------- Min-K% Prob
def _finite(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def logprobs_from_echo(choice, n_prompt_tokens):
    lp = (choice.get("logprobs") or {}).get("token_logprobs") or []
    if n_prompt_tokens:
        lp = lp[:n_prompt_tokens]
    return [float(v) for v in lp if _finite(v)]


def logprobs_from_prompt_logprobs(choice):
    out = []
    for entry in choice.get("prompt_logprobs") or []:
        if not entry:
            continue
        values = list(entry.values())
        if len(values) != 1:
            raise ValueError("expected exactly one logprob per prompt token (prompt_logprobs=0)")
        v = values[0].get("logprob") if isinstance(values[0], dict) else values[0]
        if _finite(v):
            out.append(float(v))
    return out


def min_k_prob(token_logprobs, k=MINK_K):
    """Mean of the lowest k fraction of token log probabilities (higher = more familiar)."""
    values = sorted(v for v in token_logprobs if _finite(v))
    if not values:
        return None
    n = max(1, int(math.ceil(len(values) * k)))
    return float(np.mean(values[:n]))


def mink_call(client, model, text, method):
    if method == "echo":
        r = client.completions.create(model=model, prompt=text, max_tokens=1, temperature=0,
                                      echo=True, logprobs=0)
        d = r.model_dump()
        n = (d.get("usage") or {}).get("prompt_tokens")
        return logprobs_from_echo(d["choices"][0], n), d.get("model")
    r = client.completions.create(model=model, prompt=text, max_tokens=1, temperature=0,
                                  extra_body={"prompt_logprobs": 0})
    d = r.model_dump()
    return logprobs_from_prompt_logprobs(d["choices"][0]), d.get("model")


MINK_MAX_FAIL_SHARE = 0.10   # a model failing on more items than this is excluded from P4


def run_minkprob(client, out_dir=CONTAM_DIR, limit=None, delay=C.DELAY_SECONDS):
    """Score every item for every model. A per-item API failure (e.g. Gemma's NaN error)
    is recorded and skipped; it never stops the run. Exclusion is decided in analysis."""
    out_dir = Path(out_dir)
    output = out_dir / "mink_results.jsonl"
    items = [i for g in GROUPS for i in read_jsonl(out_dir / f"items_{g}.jsonl")]
    done = {(r["model"], r["uid"]) for r in read_jsonl(output) if r.get("uid")}
    count = 0
    for model in MINK_MODELS:
        for item in items:
            if (model, item["uid"]) in done:
                continue
            if limit is not None and count >= limit:
                return count
            record, errors = None, []
            for method in ("echo", "prompt_logprobs"):
                for attempt in range(1, C.MAX_TRANSPORT_RETRIES + 2):
                    try:
                        lps, served = mink_call(client, model, item["text"], method)
                        if served and served != model:
                            raise SystemExit(f"asked for {model}, got {served}; stopped")
                        if not lps:
                            raise ValueError("no finite token log probabilities returned")
                        record = {"model": model, "uid": item["uid"], "group": item["group"],
                                  "method": method, "n_tokens": len(lps),
                                  "min_k_prob": min_k_prob(lps), "mean_logprob": float(np.mean(lps)),
                                  "timestamp_utc": utc_now()}
                        break
                    except SystemExit:
                        raise
                    except Exception as error:  # noqa: BLE001
                        errors.append(f"{method}#{attempt}: {type(error).__name__}: {str(error)[:150]}")
                        transient = getattr(error, "status_code", None) in {408, 429, 500, 502, 503, 504} \
                            or type(error).__name__ in {"APIConnectionError", "APITimeoutError"}
                        if transient and attempt <= C.MAX_TRANSPORT_RETRIES:
                            time.sleep(2.0 * 2 ** (attempt - 1))
                            continue
                        break
                if record:
                    break
            if record is None:
                record = {"model": model, "uid": item["uid"], "group": item["group"],
                          "failed": True, "errors": errors[-2:], "timestamp_utc": utc_now()}
            append_jsonl(output, record)
            count += 1
            if count % 100 == 0:
                print(f"minkprob progress: {count}", flush=True)
            if delay:
                time.sleep(delay)
    return count


# ------------------------------------------------------------------ analysis
def boot_mean_ci(values, rng, n_boot=C.N_BOOTSTRAP):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"),) * 3
    boot = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def boot_diff_ci(a, b, rng, n_boot=C.N_BOOTSTRAP):
    """Independent-sample bootstrap of mean(a) - mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return (float("nan"),) * 3
    ba = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    bb = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    d = ba - bb
    return float(a.mean() - b.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def auc(pos, neg):
    """P(score of a random pos > random neg), ties count half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def boot_auc_ci(pos, neg, rng, n_boot=2000):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return (float("nan"),) * 3
    vals = [auc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))])
            for _ in range(n_boot)]
    return auc(pos, neg), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def p1_scores(out_dir):
    """Per (group, model): list of guided-minus-general ROUGE-L per item, plus review rows."""
    scores, review = {}, []
    sources = []
    stage1 = {r["request_id"]: r for r in stage1_all(out_dir)}
    for group in ("olid_train", "fresh"):
        items = read_jsonl(Path(out_dir) / f"items_{group}.jsonl")
        sources.append((group, items, lambda m, uid, kind, s=stage1: s.get(f"p1:{m}:{uid}:{kind}")))
    main_dir = C.stage_dir("main")
    old = {r["request_id"]: r for r in read_jsonl(main_dir / "probe_predictions.jsonl")}
    test_items = read_jsonl(Path(out_dir) / "items_olid_test.jsonl")
    sources.append(("olid_test", test_items,
                    lambda m, uid, kind, s=old: s.get(f"probe:{m}:{uid.split(':', 1)[1]}:{kind}")))
    for group, items, get in sources:
        for model in C.MODELS:
            diffs = []
            for item in items:
                pair = {}
                for kind in ("general", "guided"):
                    rec = get(model, item["uid"], kind)
                    if rec and rec.get("compliant"):
                        completion = clean_completion(rec["raw_response"])
                        pair[kind] = rouge_l_f1(completion, item["reference"])
                        if pair[kind] >= C.PROBE_REVIEW_ROUGE_L:
                            review.append({"group": group, "model": model, "uid": item["uid"],
                                           "kind": kind, "rouge_l": round(pair[kind], 3),
                                           "prefix": item["prefix"], "reference": item["reference"],
                                           "completion": completion})
                if len(pair) == 2:
                    diffs.append(pair["guided"] - pair["general"])
            if diffs:
                scores[(group, model)] = diffs
    return scores, review


def p3_scores(out_dir):
    """Per (group, model): list of correct(cue) - correct(neutral) per item."""
    stage1 = {r["request_id"]: r for r in stage1_all(out_dir)}
    scores = {}
    for group in GROUPS:
        items = read_jsonl(Path(out_dir) / f"items_{group}.jsonl")
        for model in C.MODELS:
            diffs, acc = [], {"neutral": [], "cue": []}
            for item in items:
                pair = {}
                for kind in ("neutral", "cue"):
                    rec = stage1.get(f"p3:{model}:{item['uid']}:{kind}")
                    if rec is None:
                        continue
                    label = C.LABELS.get(rec.get("parsed_label")) if rec.get("compliant") else None
                    pair[kind] = int(label == item["label"])  # invalid counts as wrong
                if len(pair) == 2:
                    diffs.append(pair["cue"] - pair["neutral"])
                    acc["neutral"].append(pair["neutral"])
                    acc["cue"].append(pair["cue"])
            if diffs:
                scores[(group, model)] = {"diffs": diffs, "acc_neutral": float(np.mean(acc["neutral"])),
                                          "acc_cue": float(np.mean(acc["cue"]))}
    return scores


def p4_scores(out_dir):
    records = [r for r in read_jsonl(Path(out_dir) / "mink_results.jsonl") if r.get("uid")]
    out, attempted, failed = {}, Counter(), Counter()
    for r in records:
        attempted[r["model"]] += 1
        if r.get("failed") or r.get("min_k_prob") is None:
            failed[r["model"]] += 1
            continue
        out.setdefault((r["group"], r["model"]), []).append(r["min_k_prob"])
    excluded = sorted(m for m in attempted if failed[m] / attempted[m] > MINK_MAX_FAIL_SHARE)
    out = {k: v for k, v in out.items() if k[1] not in excluded}
    return out, excluded, {m: (failed[m], attempted[m]) for m in attempted}


def p5_scores(out_dir):
    keys = {k["uid"]: k for k in read_jsonl(Path(out_dir) / "quiz_key.jsonl")}
    out, letters, by_position = {}, {}, {}
    sources = [("quiz_predictions.jsonl", parse_letter, "")]
    if (Path(out_dir) / "quiz_long_predictions.jsonl").exists():
        sources.append(("quiz_long_predictions.jsonl", parse_letter_long, LONG_SUFFIX))
    for filename, parser, suffix in sources:
        for r in read_jsonl(Path(out_dir) / filename):
            key = keys.get(r["row_id"])
            if not key or not key.get("quiz_valid"):
                continue
            letter = parser(r.get("raw_response"))
            correct = int(letter == key["answer"])
            cell = (key["group"], r["model"] + suffix)
            out.setdefault(cell, []).append(correct)
            letters.setdefault(cell, Counter())[letter or "invalid"] += 1
            by_position.setdefault(cell, {}).setdefault(key["answer"], []).append(correct)
    return out, letters, by_position


LONG_SUFFIX = "+long"


def short_name(model):
    if model.endswith(LONG_SUFFIX):
        return C.MODEL_SHORT[model[:-len(LONG_SUFFIX)]] + " (256-token answers)"
    return C.MODEL_SHORT[model]


def analyse(out_dir=CONTAM_DIR):
    from scipy.stats import binomtest

    out_dir = Path(out_dir)
    res = out_dir / "results"
    res.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([C.BOOTSTRAP_SEED, 77])
    present = [g for g in GROUPS if (out_dir / f"items_{g}.jsonl").exists()]
    rows = []
    lines = ["# Contamination evidence battery (generated tables, no interpretation)", "",
             "A probe that finds nothing means \"no evidence detected\", not \"not in the training data\".",
             f"Groups present: {', '.join(present)}.", ""]

    # P1
    p1, review = p1_scores(out_dir)
    lines += ["## P1 Guided minus general completion (ROUGE-L)", "",
              "| Group | Model | n | Mean diff [95% CI] |", "|---|---|---:|---|"]
    for (group, model), d in sorted(p1.items()):
        m, lo, hi = boot_mean_ci(d, rng)
        rows.append({"probe": "P1", "group": group, "model": model, "n": len(d), "estimate": m,
                     "ci_low": lo, "ci_high": hi})
        lines.append(f"| {group} | {C.MODEL_SHORT[model]} | {len(d)} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    lines += _contrasts("P1", p1, rows, rng, lambda v: v)

    # P3
    p3 = p3_scores(out_dir)
    lines += ["", "## P3 Dataset-cue label probe (accuracy with OLID cue minus matched neutral prompt)", "",
              "| Group | Model | n | Acc neutral | Acc cue | Cue − neutral [95% CI] |",
              "|---|---|---:|---:|---:|---|"]
    for (group, model), v in sorted(p3.items()):
        m, lo, hi = boot_mean_ci(v["diffs"], rng)
        rows.append({"probe": "P3", "group": group, "model": model, "n": len(v["diffs"]),
                     "estimate": m, "ci_low": lo, "ci_high": hi,
                     "acc_neutral": v["acc_neutral"], "acc_cue": v["acc_cue"]})
        lines.append(f"| {group} | {C.MODEL_SHORT[model]} | {len(v['diffs'])} | {v['acc_neutral']:.3f} | "
                     f"{v['acc_cue']:.3f} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    lines += _contrasts("P3", {k: v["diffs"] for k, v in p3.items()}, rows, rng, lambda v: v)

    # P4
    p4, excluded, fails = p4_scores(out_dir)
    lines += ["", f"## P4 Min-K% Prob (k = {int(MINK_K * 100)}%; higher = more familiar)", ""]
    lines.append("API failures per model: " + ", ".join(
        f"{C.MODEL_SHORT[m]} {f}/{a}" for m, (f, a) in fails.items()) + ".")
    if excluded:
        lines.append(f"Excluded (more than {int(MINK_MAX_FAIL_SHARE * 100)}% of items failed, so the "
                     f"scored items are not a random subset): "
                     f"{', '.join(C.MODEL_SHORT[m] for m in excluded)}.")
    lines.append("")
    lines += ["| Model | Comparison | AUC [95% CI] | Note |", "|---|---|---|---|"]
    for model in C.MODELS:
        for pos, neg, note in (("olid_test", "olid_train", "same era; no time confound"),
                               ("olid_test", "fresh", "time-shift confound (Duan et al. 2024): suggestive only"),
                               ("olid_train", "fresh", "time-shift confound: suggestive only")):
            if (pos, model) in p4 and (neg, model) in p4:
                a, lo, hi = boot_auc_ci(p4[(pos, model)], p4[(neg, model)], rng)
                rows.append({"probe": "P4", "group": f"{pos} vs {neg}", "model": model,
                             "n": len(p4[(pos, model)]) + len(p4[(neg, model)]),
                             "estimate": a, "ci_low": lo, "ci_high": hi})
                lines.append(f"| {C.MODEL_SHORT[model]} | {pos} vs {neg} | {a:.3f} [{lo:.3f}, {hi:.3f}] | {note} |")
    for (group, model), v in sorted(p4.items()):
        rows.append({"probe": "P4_mean", "group": group, "model": model, "n": len(v),
                     "estimate": float(np.mean(v)), "ci_low": None, "ci_high": None})

    # P5
    p5, letters, by_position = p5_scores(out_dir)
    lines += ["", "## P5 Data Contamination Quiz (pick the original among 4; chance = 0.25)", "",
              "| Group | Model | n | Accuracy [95% CI] | Binomial p (> 0.25) | Picks A/B/C/D/invalid |",
              "|---|---|---:|---|---:|---|"]
    for (group, model), v in sorted(p5.items()):
        m, lo, hi = boot_mean_ci(v, rng)
        p = binomtest(int(sum(v)), len(v), 0.25, alternative="greater").pvalue
        c = letters[(group, model)]
        rows.append({"probe": "P5", "group": group, "model": model, "n": len(v), "estimate": m,
                     "ci_low": lo, "ci_high": hi, "p_greater_than_chance": float(p)})
        lines.append(f"| {group} | {short_name(model)} | {len(v)} | {m:.3f} [{lo:.3f}, {hi:.3f}] | "
                     f"{p:.4f} | {c['A']}/{c['B']}/{c['C']}/{c['D']}/{c['invalid']} |")
    lines += ["", "P5 diagnostics: accuracy when the original is at position A/B/C/D "
              "(a model that favours one letter scores high only there), and accuracy among valid answers.", "",
              "| Group | Model | Acc if A | Acc if B | Acc if C | Acc if D | Acc among valid |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for cell in sorted(by_position):
        pos = by_position[cell]
        valid = sum(v for k, v in letters[cell].items() if k != "invalid")
        acc_valid = sum(p5[cell]) / valid if valid else float("nan")
        cols = " | ".join(f"{np.mean(pos[L]):.2f} (n={len(pos[L])})" if L in pos else "n/a" for L in LETTERS)
        lines.append(f"| {cell[0]} | {short_name(cell[1])} | {cols} | {acc_valid:.3f} |")
    lines += _contrasts("P5", p5, rows, rng, lambda v: v)

    qmeta = out_dir / "quiz_meta.json"
    if qmeta.exists():
        vb = read_json(qmeta)["valid_by_group"]
        lines += ["", "Quiz items with 3 valid paraphrases: " +
                  ", ".join(f"{g} {v[0]}/{v[1]}" for g, v in vb.items() if v[1])]
    lines += ["", f"{len(review)} completions with ROUGE-L ≥ {C.PROBE_REVIEW_ROUGE_L} listed in "
              "contam_review_CONTAINS_TWEET_TEXT.csv (gitignored) for manual review."]

    _write_csv(res / "contam_results.csv", rows)
    _write_csv(res / "contam_review_CONTAINS_TWEET_TEXT.csv", review)
    (res / "contam_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _contrasts(probe, scores, rows, rng, f):
    lines = ["", f"{probe} group contrasts (independent bootstrap, difference in means):", "",
             "| Model | Contrast | Difference [95% CI] |", "|---|---|---|"]
    added = False
    models = list(C.MODELS) + sorted({m for _, m in scores if m.endswith(LONG_SUFFIX)})
    for model in models:
        for a, b in (("olid_test", "olid_train"), ("olid_test", "fresh"), ("olid_train", "fresh")):
            if (a, model) in scores and (b, model) in scores:
                d, lo, hi = boot_diff_ci(f(scores[(a, model)]), f(scores[(b, model)]), rng)
                rows.append({"probe": f"{probe}_contrast", "group": f"{a} - {b}", "model": model,
                             "n": len(scores[(a, model)]) + len(scores[(b, model)]),
                             "estimate": d, "ci_low": lo, "ci_high": hi})
                lines.append(f"| {short_name(model)} | {a} − {b} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
                added = True
    return lines if added else []


def _write_csv(path, rows):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--groups", nargs="+", choices=GROUPS, required=True)
    r = sub.add_parser("run")
    r.add_argument("--part", choices=["stage1", "quiz", "quiz_long"], required=True)
    r.add_argument("--groups", nargs="+", choices=GROUPS, default=["olid_train", "olid_test"],
                   help="which stage1 manifest (as given to prepare)")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--limit", type=int)
    sub.add_parser("quiz-prepare")
    sub.add_parser("quiz-long-prepare")
    m = sub.add_parser("minkprob")
    gm = m.add_mutually_exclusive_group(required=True)
    gm.add_argument("--all", action="store_true")
    gm.add_argument("--limit", type=int)
    sub.add_parser("analyze")
    args = parser.parse_args()

    if args.step == "prepare":
        meta = prepare(args.groups)
        print("CONTAM_PREPARE_OK", {"groups": meta["groups"], "n_requests": meta["n_requests"],
                                    "items": {g: s["n_items"] for g, s in meta["sources"].items()}})
    elif args.step == "run":
        from api import make_client
        from run import run
        if args.part == "stage1":
            tag = "_".join(args.groups)
            meta = read_json(CONTAM_DIR / f"stage1_meta_{tag}.json")
            manifest = CONTAM_DIR / meta["requests_file"]
            out, att = CONTAM_DIR / f"stage1_predictions_{tag}.jsonl", CONTAM_DIR / f"stage1_attempts_{tag}.jsonl"
        else:
            name = args.part
            meta = read_json(CONTAM_DIR / f"{name}_meta.json")
            manifest = CONTAM_DIR / f"{name}_requests.jsonl"
            out, att = CONTAM_DIR / f"{name}_predictions.jsonl", CONTAM_DIR / f"{name}_attempts.jsonl"
        summary = run(manifest, out, att, make_client(), limit=None if args.all else args.limit,
                      request_logprobs=False, expected_manifest_sha256=meta["requests_file_sha256"])
        print("RUN_SESSION_DONE", summary)
        if summary["remaining"] == 0:
            print("ALL_REQUESTS_COMPLETE")
    elif args.step == "quiz-prepare":
        meta = quiz_prepare()
        print("QUIZ_PREPARE_OK", {k: meta[k] for k in ("n_items", "n_valid", "valid_by_group", "n_requests")})
    elif args.step == "quiz-long-prepare":
        meta = quiz_long_prepare()
        print("QUIZ_LONG_PREPARE_OK", {k: meta[k] for k in ("models", "n_requests")})
    elif args.step == "minkprob":
        from api import make_client
        n = run_minkprob(make_client(), limit=None if args.all else args.limit)
        print("MINKPROB_SESSION_DONE", {"scored_this_session": n})
    else:
        analyse()
        print("CONTAM_ANALYSIS_OK ->", CONTAM_DIR / "results" / "contam_summary.md")


if __name__ == "__main__":
    main()
