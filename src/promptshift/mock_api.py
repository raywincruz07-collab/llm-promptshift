"""Offline stand-in for the maKI API, used for dry runs and tests.

It imitates the OpenAI client's response shape and can plant known effects
(wording flips, repeat noise, a definition effect, failures) so the analysis
can be checked against a known answer. It never contacts the network.
"""

import hashlib
import random
import re
from types import SimpleNamespace

POST_PATTERN = re.compile(r'Post: "(.*)"\n\n', re.DOTALL)
PREFIX_PATTERN = re.compile(r"First part: (.*)\nContinuation:", re.DOTALL)


class APIConnectionError(Exception):
    """Same class name as the openai SDK's, so it is treated as transport."""


def _u(*parts) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / 16 ** 12


class _Completions:
    def __init__(self, client):
        self._client = client

    def create(self, **kwargs):
        return self._client._create(**kwargs)


class MockClient:
    def __init__(self, truth=None, seed=0, recall=None, false_positive=None,
                 definition_recall_gain=None, definition_fp_change=None,
                 wording_flip=0.04, repeat_noise=0.01, noncompliance=0.0,
                 transport_failure=0.0, memory=None, memorise_rate=None,
                 general_memorise_rate=0.02, models=None):
        self.truth = truth or {}
        self.memory = memory or {}
        self.recall = recall or {"gemma4-26b": 0.40, "qwen3.6-36b": 0.40, "llama-3.3-70b": 0.78}
        self.false_positive = false_positive or {"gemma4-26b": 0.05, "qwen3.6-36b": 0.04, "llama-3.3-70b": 0.17}
        self.definition_recall_gain = definition_recall_gain or {}
        self.definition_fp_change = definition_fp_change or {}
        self.wording_flip = wording_flip
        self.repeat_noise = repeat_noise
        self.noncompliance = noncompliance
        self.transport_failure = transport_failure
        self.memorise_rate = memorise_rate or {}
        self.general_memorise_rate = general_memorise_rate
        self.model_ids = models or ["gemma4-26b", "qwen3.6-36b", "llama-3.3-70b"]
        self._rng = random.Random(seed)
        self._calls = 0
        self.chat = SimpleNamespace(completions=_Completions(self))
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id=m) for m in self.model_ids])
        )

    def _respond(self, model, content, finish="stop"):
        self._calls += 1
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message, finish_reason=finish, logprobs=None)
        usage = SimpleNamespace(prompt_tokens=50, completion_tokens=3, total_tokens=53)
        return SimpleNamespace(
            id=f"mock-{self._calls}", model=model, system_fingerprint=None,
            choices=[choice], usage=usage,
        )

    def _create(self, model, messages, temperature, max_tokens, **_):
        if self._rng.random() < self.transport_failure:
            raise APIConnectionError("simulated connection reset")
        prompt = messages[-1]["content"]
        if "Continuation:" in prompt:
            return self._complete(model, prompt)
        if self._rng.random() < self.noncompliance:
            return self._respond(model, "I think this post is offensive.")

        match = POST_PATTERN.search(prompt)
        text = match.group(1) if match else prompt
        has_definition = "Use these definitions:" in prompt
        wording = prompt.split("\n", 1)[0]
        gold = self.truth.get(text, 0)

        if gold == 1:
            p = self.recall[model] + (self.definition_recall_gain.get(model, 0) if has_definition else 0)
        else:
            p = self.false_positive[model] + (self.definition_fp_change.get(model, 0) if has_definition else 0)
        label = _u(model, text, "base") < p
        if not wording.startswith("Does the following") and _u(model, text, has_definition, wording) < self.wording_flip:
            label = not label
        if self._rng.random() < self.repeat_noise:
            label = not label
        return self._respond(model, "OFFENSIVE" if label else "NOT_OFFENSIVE")

    def _complete(self, model, prompt):
        match = PREFIX_PATTERN.search(prompt)
        prefix = match.group(1).strip() if match else ""
        guided = "TweetEval" in prompt
        rate = self.memorise_rate.get(model, 0.0) if guided else self.general_memorise_rate
        if prefix in self.memory and _u(model, prefix, guided) < rate:
            return self._respond(model, self.memory[prefix])
        filler = ["and", "then", "we", "all", "go", "home", "today", "really", "think", "so"]
        words = [filler[int(_u(model, prefix, guided, i) * len(filler))] for i in range(8)]
        return self._respond(model, " ".join(words))
