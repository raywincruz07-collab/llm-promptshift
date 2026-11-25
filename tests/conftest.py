"""Guard: the test suite must never change files in protocol/ (the study's frozen record)."""

import hashlib
from pathlib import Path

import pytest

PROTOCOL = Path(__file__).resolve().parents[1] / "protocol"


def _snapshot():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PROTOCOL.glob("*")) if p.is_file()}


@pytest.fixture(autouse=True, scope="session")
def protocol_untouched():
    before = _snapshot()
    yield
    after = _snapshot()
    assert after == before, f"tests changed protocol/ files: {sorted(set(after.items()) ^ set(before.items()))}"
