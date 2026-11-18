"""Preflight check before any study request.

    python src/promptshift/check_api.py

Lists the endpoint's models, confirms the three aliases exist, sends ONE
request per model using a neutral test sentence (not a study post), and
tests whether log probabilities are supported. Results go to
data/preflight.json. These calls are not part of the study.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
from api import make_client  # noqa: E402
from common import utc_now, write_json  # noqa: E402
from prompts import render  # noqa: E402

TEST_TEXT = "The weather in Mannheim was pleasant today."


def main(mock=False):
    client = make_client(mock=mock)
    available = sorted(model.id for model in client.models.list().data)
    report = {"checked_utc": utc_now(), "available_models": available, "models": {}}
    for model in C.MODELS:
        entry = {"listed": model in available}
        prompt = render("W0", TEST_TEXT)
        base = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": C.TEMPERATURE,
            "max_tokens": C.MAX_TOKENS_CLASSIFY,
        }
        try:
            response = client.chat.completions.create(**base)
            entry.update(
                ok=True,
                response_model=response.model,
                reply=response.choices[0].message.content,
                finish_reason=response.choices[0].finish_reason,
                system_fingerprint=getattr(response, "system_fingerprint", None),
            )
        except Exception as error:  # noqa: BLE001
            entry.update(ok=False, error=f"{type(error).__name__}: {error}")
        try:
            response = client.chat.completions.create(**base, logprobs=True, top_logprobs=5)
            entry["logprobs_supported"] = bool(
                getattr(response.choices[0], "logprobs", None)
                and response.choices[0].logprobs.content
            )
        except Exception as error:  # noqa: BLE001
            entry["logprobs_supported"] = False
            entry["logprobs_error"] = f"{type(error).__name__}: {error}"
        report["models"][model] = entry
        print(model, entry)

    write_json(C.DATA_DIR / ("preflight_mock.json" if mock else "preflight.json"), report)
    ok = all(e.get("listed") and e.get("ok") and e.get("response_model") == m
             for m, e in report["models"].items())
    print("PREFLIGHT_OK" if ok else "PREFLIGHT_PROBLEM: read data/preflight.json")
    if ok:
        supported = all(e["logprobs_supported"] for e in report["models"].values())
        print(f"Log probabilities supported for all models: {supported}. "
              "Set REQUEST_LOGPROBS in config.py accordingly before registering.")


if __name__ == "__main__":
    main(mock="--mock" in sys.argv)
