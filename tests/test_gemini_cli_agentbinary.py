"""Focused Gemini CLI provisioning tests."""

from unittest.mock import AsyncMock, Mock, patch

import anyio
import pytest

from inspect_swe._gemini_cli import agentbinary


def test_sandbox_version_uses_the_attached_binary_without_network_resolution() -> None:
    sandbox = Mock()
    with (
        patch.object(agentbinary, "detect_sandbox_platform", AsyncMock(return_value="linux-x64")),
        patch.object(agentbinary, "ensure_node_available", AsyncMock(return_value="/node")),
        patch.object(
            agentbinary,
            "_sandbox_gemini_binary",
            AsyncMock(return_value="/usr/local/bin/gemini"),
        ) as installed_binary,
        patch.object(
            agentbinary,
            "resolve_gemini_version",
            AsyncMock(side_effect=AssertionError("sandbox must not resolve latest")),
        ),
    ):
        binary, node = anyio.run(agentbinary.ensure_gemini_cli_setup, sandbox, "sandbox", "agent")

    assert (binary, node) == ("/usr/local/bin/gemini", "/node")
    installed_binary.assert_awaited_once_with(sandbox, "/node", "agent")


def test_sandbox_version_fails_loudly_when_the_attached_binary_is_missing() -> None:
    sandbox = Mock()
    with (
        patch.object(agentbinary, "detect_sandbox_platform", AsyncMock(return_value="linux-x64")),
        patch.object(agentbinary, "ensure_node_available", AsyncMock(return_value="/node")),
        patch.object(
            agentbinary,
            "_sandbox_gemini_binary",
            AsyncMock(side_effect=RuntimeError("attached Gemini CLI binary is unavailable")),
        ),
        patch.object(
            agentbinary,
            "resolve_gemini_version",
            AsyncMock(side_effect=AssertionError("sandbox must not resolve latest")),
        ),
        pytest.raises(RuntimeError, match="attached Gemini CLI binary is unavailable"),
    ):
        anyio.run(agentbinary.ensure_gemini_cli_setup, sandbox, "sandbox", "agent")
