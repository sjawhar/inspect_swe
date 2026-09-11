"""Antigravity CLI invocation deadline.

This module is deliberately not self-contained. `run_unattended_agent` and the fake
sandboxes it drives both arrive from `fix/codex-exec-timeout`, which composes with this
branch at the release cut; in isolation this file does not import. Reusing that
scaffolding rather than copying it is the point — the bug being fixed here is one
bounding helper that six agents were supposed to share and five did.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib import import_module
from typing import final
from unittest.mock import patch

import anyio
import pytest
from inspect_ai.agent import AgentState
from inspect_ai.util._sandbox import exec_remote as core_exec_remote
from inspect_swe._antigravity_cli.antigravity_cli import antigravity_cli

from tests.test_cli_exec_timeout import (
    _agent_state,
    _CoreKillSandbox,
    _ensure_binary,
    _FakeStore,
    _resolve_cwd,
)


@final
class _AntigravityBridge:
    """What the shared codex fake provides, plus the attribute the CLI agent reads.

    `_FakeBridge` is `@final`, so this mirrors its fields rather than extending it.
    `build_antigravity_mcp_config` calls `.get` on `bridged_tools`, so the real shape
    is a mapping and not a list -- worth getting right here even though this test
    configures no MCP servers, because the next test to reuse it will.
    """

    state: AgentState
    port: int
    mcp_server_configs: list[object]
    bridged_tools: dict[str, dict[str, object]]

    def __init__(self, state: AgentState) -> None:
        self.state = state
        self.port = 3001
        self.mcp_server_configs = []
        self.bridged_tools = {}


@asynccontextmanager
async def _antigravity_bridge(
    state: AgentState, **_kwargs: object
) -> AsyncGenerator[_AntigravityBridge, None]:
    yield _AntigravityBridge(state)


def _run_antigravity_cli(
    sandbox: _CoreKillSandbox,
    *,
    exec_timeout: float | None = None,
) -> AgentState:
    antigravity_cli_module = import_module(
        "inspect_swe._antigravity_cli.antigravity_cli"
    )

    with (
        patch.object(
            antigravity_cli_module, "sandbox_agent_bridge", _antigravity_bridge
        ),
        patch.object(antigravity_cli_module, "sandbox_env", return_value=sandbox),
        # Patched together with the installer: the production call evaluates
        # `antigravity_cli_binary_source()` before `ensure_agent_binary_installed`
        # runs, and that factory creates ~/.cache/inspect_swe/... on the host. The
        # patched installer ignores its argument, so a sentinel is enough.
        patch.object(
            antigravity_cli_module, "antigravity_cli_binary_source", lambda: object()
        ),
        patch.object(
            antigravity_cli_module, "ensure_agent_binary_installed", _ensure_binary
        ),
        patch.object(antigravity_cli_module, "resolve_agent_cwd", _resolve_cwd),
        patch.object(antigravity_cli_module, "store", return_value=_FakeStore()),
    ):
        agent = (
            antigravity_cli()
            if exec_timeout is None
            else antigravity_cli(exec_timeout=exec_timeout)
        )
        return anyio.run(agent, _agent_state())


def test_antigravity_cli_exec_timeout_kills_real_core_process() -> None:
    sandbox = _CoreKillSandbox()

    try:
        with (
            patch.object(
                core_exec_remote, "exec_model_request", sandbox.exec_model_request
            ),
            pytest.raises(
                RuntimeError,
                match="Antigravity CLI execution timed out after 0.01 seconds",
            ),
        ):
            _ = _run_antigravity_cli(sandbox, exec_timeout=0.01)

        assert sandbox.kill_rpc_count == 1
        assert sandbox.exit_code == -9
    finally:
        sandbox.cleanup()
