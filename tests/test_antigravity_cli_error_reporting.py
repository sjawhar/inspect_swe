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

from inspect_swe._antigravity_cli.antigravity_cli import _clean_antigravity_error

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
