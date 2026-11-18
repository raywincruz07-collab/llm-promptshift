"""Project configuration. Every value here is part of the pre-registered protocol.

Commit this file and record the commit SHA in the AsPredicted registration
BEFORE any main-stage model call.

Code drafted with AI assistance (Claude, Anthropic); the team reviewed it and
is responsible for it (course AI-use policy).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Data: TweetEval "offensive" (OLID level A), pinned to an exact revision.
DATASET_REPO = "cardiffnlp/tweet_eval"
DATASET_REVISION = "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
SPLIT_SHA256 = {
    "validation": "1161b4e6cc7cbdaa667d4fc131f9be0ea3db755616126329dda7eb521cd88c36",
    "test": "15b87adaf1850d8e3e104be6240255813d48d776940be546ad353fee9fcfb099",
}
SPLIT_LABEL_COUNTS = {
    "validation": {0: 865, 1: 459},
    "test": {0: 620, 1: 240},
}

# Stages and what each data split is for:
#   pilot -> 40 random VALIDATION posts. Only checks that every prompt yields
#            a parseable label (compliance). Never used for any hypothesis.
#   main  -> the full official TEST split (860 posts). All reported results.
STAGES = {
    "pilot": {"split": "validation", "n_posts": 40},
    "main": {"split": "test", "n_posts": None},
}

# Primary H2 metric: "macro_f1" (all three models, two-sided).
# Alternative kept for transparency: "offensive_recall" (Gemma, Qwen, one-sided).
PRIMARY_H2 = "macro_f1"

# Request log probabilities? Set after check_api.py; never change mid-run.
REQUEST_LOGPROBS = True

SHUFFLE_SEED = 20261123
BOOTSTRAP_SEED = 20261124
PROBE_SEED = 20261125
PILOT_SEED = 20261126

MODELS = ["gemma4-26b", "qwen3.6-36b", "llama-3.3-70b"]
MODEL_SHORT = {
    "gemma4-26b": "Gemma",
    "qwen3.6-36b": "Qwen",
    "llama-3.3-70b": "Llama",
}

CONDITIONS = ["W0", "W1", "W2", "D0", "D1", "D2", "R0", "RD0"]
WORDING_OF = {
    "W0": "P0", "W1": "P1", "W2": "P2",
    "D0": "P0", "D1": "P1", "D2": "P2",
    "R0": "P0", "RD0": "P0",
}
HAS_DEFINITION = {c: c.startswith("D") or c == "RD0" for c in CONDITIONS}
REPEAT_OF = {"R0": "W0", "RD0": "D0"}

API_BASE_URL = "https://maki.uni-mannheim.de/v1"
API_KEY_ENV = "MAKI_API_KEY"
TEMPERATURE = 0
MAX_TOKENS_CLASSIFY = 16
MAX_TOKENS_PROBE = 64
DELAY_SECONDS = 0.25
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_TRANSPORT_RETRIES = 3
STOP_AFTER_CONSECUTIVE_TRANSPORT_FAILURES = 10

LABELS = {"NOT_OFFENSIVE": 0, "OFFENSIVE": 1}

N_BOOTSTRAP = 10_000
ALPHA = 0.05
SESOI = {
    "H1": 0.01,
    "macro_f1": 0.02,
    "offensive_recall": 0.05,
}

PROBE_N_POSTS = 200
PROBE_MIN_CONTENT_TOKENS = 12
PROBE_PLACEHOLDER_TOKENS = {"@USER", "URL"}
PROBE_REVIEW_ROUGE_L = 0.7

ANNOTATION_N_FLIP = 50
ANNOTATION_N_STABLE = 50
ANNOTATION_SEED = 20261127

DATA_DIR = ROOT / "data"          # gitignored: contains post text
CACHE_DIR = DATA_DIR / "source"
PROTOCOL_DIR = ROOT / "protocol"  # committed: hashes only, no post text


def stage_dir(stage: str) -> Path:
    return DATA_DIR / stage
