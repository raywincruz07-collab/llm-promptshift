"""API client creation, single calls and response records."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
from common import utc_now  # noqa: E402

TRANSPORT_ERROR_NAMES = {
    "APIConnectionError", "APITimeoutError", "RateLimitError",
    "InternalServerError", "ConnectError", "ConnectTimeout", "ReadTimeout",
    "RemoteProtocolError", "TimeoutError", "ConnectionError",
}
TRANSPORT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class ModelMismatchError(RuntimeError):
    """The API answered with a different model than requested."""


class StopRuleTriggered(RuntimeError):
    """A pre-specified stop rule fired."""


def is_transport_error(error: Exception) -> bool:
    if type(error).__name__ in TRANSPORT_ERROR_NAMES:
        return True
    return getattr(error, "status_code", None) in TRANSPORT_STATUS_CODES


def make_client(mock=False, mock_options=None):
    if mock:
        from mock_api import MockClient
        return MockClient(**(mock_options or {}))
    from openai import OpenAI  # only needed for live runs
    api_key = os.environ.get(C.API_KEY_ENV)
    if not api_key:
        raise SystemExit(f"Set the {C.API_KEY_ENV} environment variable first.")
    return OpenAI(
        base_url=C.API_BASE_URL,
        api_key=api_key,
        timeout=C.REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )


def call(client, request: dict, request_logprobs: bool):
    kwargs = {
        "model": request["model"],
        "messages": [{"role": "user", "content": request["prompt_text"]}],
        "temperature": C.TEMPERATURE,
        "max_tokens": request["max_tokens"],
    }
    if request_logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5
    return client.chat.completions.create(**kwargs)


def _logprob_summary(choice, limit=3):
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs else None
    if not content:
        return None
    return [
        {
            "token": item.token,
            "logprob": item.logprob,
            "top": [
                {"token": top.token, "logprob": top.logprob}
                for top in (getattr(item, "top_logprobs", None) or [])
            ],
        }
        for item in content[:limit]
    ]


def response_record(request: dict, response, latency: float, attempts: int,
                    request_logprobs: bool) -> dict:
    choice = response.choices[0]
    content = choice.message.content
    usage = getattr(response, "usage", None)
    record = {
        "request_id": request["request_id"],
        "order": request.get("order"),
        "task": request["task"],
        "sample": request.get("sample"),
        "model": request["model"],
        "row_id": request["row_id"],
        "condition": request.get("condition"),
        "prompt_sha256": request["prompt_sha256"],
        "response_model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "raw_response": content,
        "finish_reason": choice.finish_reason,
        "temperature": C.TEMPERATURE,
        "max_tokens": request["max_tokens"],
        "logprobs_requested": request_logprobs,
        "logprobs": _logprob_summary(choice) if request_logprobs else None,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "latency_seconds": round(latency, 3),
        "attempts": attempts,
        "timestamp_utc": utc_now(),
    }
    if request["task"] == "classify":
        # Strict parsing, identical to Study A: the whole reply must be a label.
        parsed = content if content in C.LABELS else None
        lenient = None
        if isinstance(content, str):
            candidate = content.strip().strip(".").upper()
            lenient = candidate if candidate in C.LABELS else None
        record["parsed_label"] = parsed
        record["parsed_label_lenient"] = lenient
        record["compliant"] = parsed is not None and choice.finish_reason == "stop"
    else:
        record["compliant"] = isinstance(content, str) and bool(content.strip())
    return record
