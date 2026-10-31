
"""Prompt templates for the IS6170 offensive-language robustness study."""

PROMPTS = {
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
        "Respond with exactly one label:\n"
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


def render_prompt(prompt_id: str, text: str) -> str:
    """Insert a social-media post into the selected prompt."""
    if prompt_id not in PROMPTS:
        raise ValueError(f"Unknown prompt ID: {prompt_id}")

    return PROMPTS[prompt_id].format(text=text)
