"""Unit tests for the Antigravity CLI agent's failure reporting.

When `agy` exits non-zero the only record of why is what this helper returns:
it becomes the RuntimeError message, which is what lands in the Inspect eval
log. Getting it wrong fails SILENTLY -- the run still errors, it just errors
without a cause, and on a large eval set there is nothing left to diagnose
from. That is what these tests exist to catch.

Measured against the real `agy` binary (1.1.20): on failure the CLI writes its
reasoning stream to stdout as ~2KB base64 Gemini thought signatures, one per
turn, and the actual reason ("Error: timeout waiting for response") to stderr.
"""

import json

import pytest
from inspect_swe._antigravity_cli.antigravity_cli import (
    _clean_antigravity_error,
    _verify_native_result,
)

REASON = "Error: timeout waiting for response"
# Shaped like what agy actually prints: base64 signature then a closing tag on
# its own line. The old filter only dropped lines STARTING with "<think", so
# nothing here was filtered.
SIGNATURE = "Ep8QCpwQAR" + "Zm9vYmFy" * 300


def test_real_reason_survives_a_stdout_full_of_thought_signatures() -> None:
    # Given: many turns' worth of opaque payload on stdout, the reason on stderr
    stdout = "\n".join(f"{SIGNATURE}\n</think>" for _ in range(25))

    # When: the failure output is cleaned for the traceback
    cleaned = _clean_antigravity_error(stdout, f"{REASON}\n")

    # Then: the reason is present and leads, rather than being truncated away.
    assert REASON in cleaned
    assert cleaned.startswith("STDERR:")
    # And: no raw signature survives to crowd it out.
    assert SIGNATURE not in cleaned


def test_opaque_payloads_are_replaced_not_merely_truncated() -> None:
    cleaned = _clean_antigravity_error(SIGNATURE, "")

    assert SIGNATURE not in cleaned
    assert "opaque payload" in cleaned


def test_ordinary_output_is_preserved_verbatim() -> None:
    cleaned = _clean_antigravity_error("a note on stdout", "a reason on stderr")

    assert "a reason on stderr" in cleaned
    assert "a note on stdout" in cleaned


def test_no_output_is_reported_as_such() -> None:
    assert _clean_antigravity_error("", "") == "Unknown error (no output)"


_RESULT_CID = "eccac0fd-d2b5-4b39-9888-175170faece0"
_OTHER_CID = "16fd2706-8baf-433b-82eb-8c7fada847da"
_NATIVE_RESULT = (
    '{"conversation_id":"eccac0fd-d2b5-4b39-9888-175170faece0",'
    '"status":"SUCCESS","response":"tool call for tool run_command\\n'
    'FINAL_NATIVE_STORE_JSON\\n","duration_seconds":4.712014882,"num_turns":1,'
    '"usage":{"input_tokens":0,"output_tokens":0,"thinking_tokens":0,'
    '"cache_read_tokens":0,"total_tokens":0}}'
)


def _result(status: str = "SUCCESS", conversation_id: str | None = _RESULT_CID) -> str:
    payload = json.loads(_NATIVE_RESULT)
    payload["conversation_id"] = conversation_id
    payload["status"] = status
    return json.dumps(payload)


def test_successful_result_for_the_bound_conversation_verifies() -> None:
    _verify_native_result(_result(), _RESULT_CID)


def test_result_naming_another_conversation_fails() -> None:
    with pytest.raises(RuntimeError, match=_OTHER_CID):
        _verify_native_result(_result(conversation_id=_OTHER_CID), _RESULT_CID)


@pytest.mark.parametrize(
    "result",
    [
        _result(status="ERROR"),
        json.dumps({"conversation_id": _RESULT_CID}),
        "",
        "not json at all",
        "{unclosed",
        f"{_result()}\n{{unclosed",
        f"{_result()}\n{_result()}",
    ],
)
def test_only_one_success_result_envelope_is_accepted(result: str) -> None:
    with pytest.raises(RuntimeError):
        _verify_native_result(result, _RESULT_CID)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_result(conversation_id=None), None),
        (_result(conversation_id=_RESULT_CID), None),
        (_result(conversation_id=None), _RESULT_CID),
        (json.dumps({"conversation_id": 42, "status": "SUCCESS"}), _RESULT_CID),
    ],
)
def test_missing_or_non_string_conversation_identity_fails(
    result: str, expected: str | None
) -> None:
    with pytest.raises(RuntimeError):
        _verify_native_result(result, expected)
