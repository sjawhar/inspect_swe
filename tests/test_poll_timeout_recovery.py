"""Every agent that opens a model bridge accepts ``poll_timeout_recovery``.

The bridge-call behavior itself is covered per agent (see
``tests/test_claude_code_model.py`` and ``tests/test_codex_config.py``). This
checks the public surface: each constructor takes the parameter keyword-only,
defaulting to ``None``, after every other named parameter.
"""

import inspect
from typing import Any, Callable, get_args, get_type_hints

import pytest
from inspect_swe import (
    antigravity,
    claude_code,
    codex_cli,
    gemini_cli,
    interactive_claude_code,
    interactive_codex_cli,
    interactive_gemini_cli,
    kimi_code,
    mini_swe_agent,
    opencode,
)
from inspect_swe.acp import ACPAgentParams

_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


@pytest.mark.parametrize(
    "constructor",
    [
        pytest.param(constructor, id=constructor.__name__)
        for constructor in (
            antigravity,
            claude_code,
            codex_cli,
            gemini_cli,
            kimi_code,
            mini_swe_agent,
            opencode,
        )
    ],
)
def test_poll_timeout_recovery_is_the_last_named_parameter(
    constructor: Callable[..., Any],
) -> None:
    named = [
        parameter
        for parameter in inspect.signature(constructor).parameters.values()
        if parameter.kind not in _VARIADIC
    ]
    last = named[-1]
    assert last.name == "poll_timeout_recovery"
    assert last.kind is inspect.Parameter.KEYWORD_ONLY
    assert last.default is None
    hint = get_type_hints(constructor)["poll_timeout_recovery"]
    assert set(get_args(hint)) == {float, type(None)}


def test_poll_timeout_recovery_is_the_last_acp_agent_param() -> None:
    hints = get_type_hints(ACPAgentParams)
    assert list(hints)[-1] == "poll_timeout_recovery"
    assert set(get_args(hints["poll_timeout_recovery"])) == {float, type(None)}


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(factory, id=factory.__name__)
        for factory in (
            interactive_claude_code,
            interactive_codex_cli,
            interactive_gemini_cli,
        )
    ],
)
def test_interactive_agents_accept_poll_timeout_recovery(
    factory: Callable[..., Any],
) -> None:
    (kwargs_type,) = get_args(get_type_hints(factory, include_extras=True)["kwargs"])
    assert "poll_timeout_recovery" in get_type_hints(kwargs_type)
