"""End-to-end coverage for an agent that spawns a subagent.

No other example delegates, so before this one nothing in the suite ever made a
real CLI write a subagent transcript -- the recorder's subagent path was tested
only against rows a person typed. That is how agent-c#19396 shipped: the first
subagent of any Claude session raised `Claude native response ID changed agent
ownership` out of the drain, while every test stayed green.

These runs assert the drain consumed a real subagent transcript, which is the
part a hand-written fixture cannot establish.
"""

from typing import Literal

import pytest
from inspect_ai.event import SpanBeginEvent, SpanEndEvent
from inspect_ai.log import EvalLog

from tests.conftest import (
    get_available_sandboxes,
    run_example,
    skip_if_no_anthropic,
    skip_if_no_docker,
)


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_claude_code_delegation(sandbox: str) -> None:
    check_delegation("claude_code", "anthropic/claude-sonnet-4-5", sandbox)


@skip_if_no_anthropic
@skip_if_no_docker
@pytest.mark.parametrize("sandbox", get_available_sandboxes())
def test_opencode_delegation(sandbox: str) -> None:
    check_delegation("opencode", "anthropic/claude-sonnet-4-5", sandbox)


def check_delegation(
    agent: Literal["claude_code", "codex_cli", "gemini_cli", "opencode"],
    model: str,
    sandbox: str,
) -> None:
    log: EvalLog = run_example("delegation", agent, model, sandbox=sandbox)[0]

    # A crash in the native drain surfaces as a sample error, which is exactly
    # what #19396 looked like in production.
    assert log.status == "success", log.error
    assert log.samples

    events = log.samples[0].events
    agent_spans = [
        event.id
        for event in events
        if isinstance(event, SpanBeginEvent) and event.id.startswith("agent-")
    ]
    assert agent_spans, "the run never spawned a subagent, so it proves nothing"

    # Every child span the drain opened must also have been closed: a subagent
    # left waiting on a response it never produced hangs here rather than
    # raising, so an open span is its own failure mode.
    ended = {event.id for event in events if isinstance(event, SpanEndEvent)}
    assert set(agent_spans) <= ended, (
        f"subagent spans left open: {set(agent_spans) - ended}"
    )
