"""The eight prompt conditions.

Three instruction wordings (P0-P2) express the same task. D-conditions insert
one fixed OLID level-A definition block after the first instruction sentence
and change nothing else. R0 and RD0 are exact copies of W0 and D0, sent as
separate requests to measure run-to-run variation.
"""

from config import CONDITIONS, HAS_DEFINITION, WORDING_OF
from common import sha256_text

BASE_PROMPTS = {
    "P0": (
        "Does the following social-media post contain offensive language?\n\n"
        'Post: "{text}"\n\n'
        "Answer with exactly one label:\n"
        "OFFENSIVE or NOT_OFFENSIVE."
    ),
    "P1": (
        "Determine whether offensive language is present in the following "
        "social-media post.\n\n"
        'Post: "{text}"\n\n'
        "Answer with exactly one label:\n"
        "OFFENSIVE or NOT_OFFENSIVE."
    ),
    "P2": (
        "Classify the following social-media post based on whether it "
        "contains offensive language.\n\n"
        'Post: "{text}"\n\n'
        "Answer with exactly one label:\n"
        "OFFENSIVE or NOT_OFFENSIVE."
    ),
}

# OLID level-A definitions, verbatim from Zampieri et al. (2019), section 2.1.
# Only the label names are replaced by the output labels the model must use.
OLID_OFFENSIVE = (
    "Posts containing any form of non-acceptable language (profanity) "
    "or a targeted offense, which can be veiled or direct."
)
OLID_NOT_OFFENSIVE = "Posts that do not contain offense or profanity."

DEFINITION_BLOCK = (
    "Use these definitions:\n"
    f"OFFENSIVE: {OLID_OFFENSIVE}\n"
    f"NOT_OFFENSIVE: {OLID_NOT_OFFENSIVE}"
)

SEPARATOR = "\n\n"


def _with_definition(template: str) -> str:
    first_sentence, rest = template.split(SEPARATOR, 1)
    assert rest.startswith('Post: "{text}"'), "unexpected template shape"
    return first_sentence + SEPARATOR + DEFINITION_BLOCK + SEPARATOR + rest


def condition_templates() -> dict:
    templates = {}
    for condition in CONDITIONS:
        base = BASE_PROMPTS[WORDING_OF[condition]]
        templates[condition] = _with_definition(base) if HAS_DEFINITION[condition] else base
    return templates


def render(condition: str, text: str) -> str:
    return condition_templates()[condition].format(text=text)


def prompt_registry() -> dict:
    templates = condition_templates()
    return {
        "definition_block": DEFINITION_BLOCK,
        "definition_block_sha256": sha256_text(DEFINITION_BLOCK),
        "conditions": {
            condition: {
                "wording": WORDING_OF[condition],
                "has_definition": HAS_DEFINITION[condition],
                "template": template,
                "template_sha256": sha256_text(template),
            }
            for condition, template in templates.items()
        },
    }
