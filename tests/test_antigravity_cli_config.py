"""Unit tests for the Antigravity CLI agent's on-disk configuration.

Both builders encode behaviour that was measured against the real `agy` binary
(1.1.20) rather than read off the documentation, and in both cases getting it
wrong fails SILENTLY -- the CLI keeps running with the setting or the server
ignored. That is what these tests exist to catch.
"""

import json
from typing import Any

import anyio
import pytest
from inspect_ai.tool._mcp._config import (
    MCPServerConfigHTTP,
    MCPServerConfigStdio,
)
from inspect_swe._antigravity_cli import agentbinary
from inspect_swe._antigravity_cli.antigravity_cli import (
    build_antigravity_mcp_config,
    build_antigravity_settings,
)


def _settings() -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(build_antigravity_settings())
    return parsed


def test_settings_select_the_direct_gemini_api_route() -> None:
    # the load-bearing key: without it the CLI blocks on OAuth sign-in and
    # never reads GEMINI_API_KEY / GOOGLE_GEMINI_BASE_URL at all.
    assert _settings()["modelProvider"] == "gemini"


@pytest.mark.parametrize(
    "key",
    ["enableTelemetry", "showTips", "showFeedbackSurvey", "enableTerminalSandbox"],
)
def test_boolean_settings_are_json_booleans(key: str) -> None:
    # These four are documented with "on"/"off" wording, but the CLI only
    # accepts real booleans: a string is dropped on load with no error and
    # without rewriting settings.json, so the file reads "off" while /config
    # still reports the default (telemetry ON).
    value = _settings()[key]
    assert isinstance(value, bool), f"{key} must be a JSON boolean, got {value!r}"
    assert value is False


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("toolPermission", "always-proceed"),
        ("artifactReviewPolicy", "always-proceed"),
        ("altScreenMode", "never"),
    ],
)
def test_enum_settings_use_the_cli_vocabulary(key: str, expected: str) -> None:
    assert _settings()[key] == expected


def test_http_mcp_servers_use_server_url() -> None:
    # The CLI's remote schema is `serverUrl`. `url`/`httpUrl` are accepted by the
    # JSON parser and then ignored, which presents as a configured server that
    # never connects.
    config = json.loads(
        build_antigravity_mcp_config(
            [
                MCPServerConfigHTTP(
                    type="http", name="agent-c", url="http://localhost:8901/mcp"
                )
            ]
        )
    )
    server = config["mcpServers"]["agent-c"]
    assert server["serverUrl"] == "http://localhost:8901/mcp"
    assert "url" not in server
    assert "httpUrl" not in server


def test_stdio_mcp_servers_keep_command_and_args() -> None:
    config = json.loads(
        build_antigravity_mcp_config(
            [
                MCPServerConfigStdio(
                    type="stdio", name="local", command="server", args=["--flag"]
                )
            ]
        )
    )
    server = config["mcpServers"]["local"]
    assert server["command"] == "server"
    assert server["args"] == ["--flag"]


def test_no_mcp_servers_still_writes_an_empty_registry() -> None:
    # The CLI reads the file unconditionally; an absent `mcpServers` object is a
    # parse error it reports as a broken configuration.
    assert json.loads(build_antigravity_mcp_config([])) == {"mcpServers": {}}


def test_binary_source_installs_under_the_agy_name() -> None:
    source = agentbinary.antigravity_cli_binary_source()
    # The release tarball's single member is named `antigravity`; the sandbox
    # binary (and the `version="sandbox"` probe) is `agy`.
    assert source.binary == "agy"
    assert source.post_download is not None


def test_binary_source_rejects_musl_platforms() -> None:
    # Only glibc assets are published. Installing the glibc binary on a musl
    # image fails at exec time with an opaque loader error, so refuse up front.
    source = agentbinary.antigravity_cli_binary_source()
    with pytest.raises(ValueError, match="Unsupported platform"):
        anyio.run(source.resolve_version, "1.1.20", "linux-x64-musl")
