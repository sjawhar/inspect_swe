"""Unit tests for the Antigravity CLI agent's on-disk configuration.

Both builders encode behaviour that was measured against the real `agy` binary
(1.1.20) rather than read off the documentation, and in both cases getting it
wrong fails SILENTLY -- the CLI keeps running with the setting or the server
ignored. That is what these tests exist to catch.
"""

import json
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.tool._mcp._config import (
    MCPServerConfigHTTP,
    MCPServerConfigStdio,
)
from inspect_swe._antigravity_cli import agentbinary
from inspect_swe._antigravity_cli.antigravity_cli import (
    _native_conversation_id,
    _NativeConversation,
    build_antigravity_mcp_config,
    build_antigravity_settings,
)
from inspect_swe._util.agentbinary import AgentBinaryVersion
from inspect_swe._util.sandbox import SandboxPlatform


def _settings(unattended: bool = True) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(
        build_antigravity_settings(unattended=unattended)
    )
    return parsed


def test_settings_select_the_direct_gemini_api_route() -> None:
    # the load-bearing key: without it the CLI blocks on OAuth sign-in and
    # never reads GEMINI_API_KEY / GOOGLE_GEMINI_BASE_URL at all.
    assert _settings()["modelProvider"] == "gemini"


def test_centaur_settings_withhold_the_approval_policies() -> None:
    centaur = _settings(unattended=False)

    assert "toolPermission" not in centaur
    assert "artifactReviewPolicy" not in centaur
    assert centaur["modelProvider"] == "gemini"
    assert centaur["enableTelemetry"] is False
    assert set(_settings()) - set(centaur) == {
        "toolPermission",
        "artifactReviewPolicy",
    }


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
            ],
            eager_tools={},
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
            ],
            eager_tools={},
        )
    )
    server = config["mcpServers"]["local"]
    assert server["command"] == "server"
    assert server["args"] == ["--flag"]


def test_no_mcp_servers_still_writes_an_empty_registry() -> None:
    # The CLI reads the file unconditionally; an absent `mcpServers` object is a
    # parse error it reports as a broken configuration.
    assert json.loads(build_antigravity_mcp_config([], eager_tools={})) == {
        "mcpServers": {}
    }


def test_stdio_cwd_is_written_as_a_string() -> None:
    config = json.loads(
        build_antigravity_mcp_config(
            [
                MCPServerConfigStdio(
                    type="stdio",
                    name="local",
                    command="server",
                    cwd=Path("/srv/mcp"),
                )
            ],
            eager_tools={},
        )
    )

    assert config["mcpServers"]["local"]["cwd"] == "/srv/mcp"


def test_bridged_tools_are_marked_eager_under_their_own_server() -> None:
    config = json.loads(
        build_antigravity_mcp_config(
            [
                MCPServerConfigHTTP(
                    type="http",
                    name="inspect-tools",
                    url="http://localhost:3001/mcp/inspect-tools",
                )
            ],
            eager_tools={"inspect-tools": ["bash", "submit"]},
        )
    )

    assert config["mcpServers"]["inspect-tools"]["tools"] == {
        "bash": {"eager": True},
        "submit": {"eager": True},
    }


def test_unbridged_servers_keep_their_lazy_tool_configuration() -> None:
    config = json.loads(
        build_antigravity_mcp_config(
            [
                MCPServerConfigHTTP(
                    type="http",
                    name="remote",
                    url="https://mcp.example.com/mcp",
                    tools=["lookup"],
                ),
                MCPServerConfigHTTP(
                    type="http",
                    name="inspect-tools",
                    url="http://localhost:3001/mcp/inspect-tools",
                ),
            ],
            eager_tools={"inspect-tools": []},
        )
    )

    assert "tools" not in config["mcpServers"]["remote"]
    assert "tools" not in config["mcpServers"]["inspect-tools"]


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


def test_downloaded_archive_is_unpacked_to_its_executable_member() -> None:
    executable = b"\x7fELF\x02\x01\x01" + b"agy" * 64
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        member = tarfile.TarInfo(name="antigravity")
        member.size = len(executable)
        tar.addfile(member, BytesIO(executable))

    source = agentbinary.antigravity_cli_binary_source()
    assert source.post_download is not None
    assert source.post_download(archive.getvalue()) == executable


_RELEASE: dict[str, Any] = {
    "assets": [
        {
            "name": "agy_cli_linux_x64.tar.gz",
            "digest": "sha256:1111",
            "browser_download_url": "https://example.com/agy_cli_linux_x64.tar.gz",
        },
        {
            "name": "agy_cli_linux_arm64.tar.gz",
            "digest": "sha256:2222",
            "browser_download_url": "https://example.com/agy_cli_linux_arm64.tar.gz",
        },
    ]
}


def _resolve(
    platform: SandboxPlatform,
    release: dict[str, Any] | None = None,
) -> AgentBinaryVersion:
    source = agentbinary.antigravity_cli_binary_source()
    with patch.object(
        agentbinary, "_fetch_release", AsyncMock(return_value=release or _RELEASE)
    ):
        return anyio.run(source.resolve_version, "1.1.27", platform)


@pytest.mark.parametrize(
    ("platform", "asset", "checksum"),
    [
        ("linux-x64", "agy_cli_linux_x64.tar.gz", "1111"),
        ("linux-arm64", "agy_cli_linux_arm64.tar.gz", "2222"),
    ],
)
def test_each_architecture_resolves_to_its_own_verified_asset(
    platform: SandboxPlatform, asset: str, checksum: str
) -> None:
    resolved = _resolve(platform)

    assert resolved.download_url.endswith(asset)
    assert resolved.expected_checksum == checksum
    assert resolved.package is False


def test_a_release_with_no_download_url_is_rejected() -> None:
    release = {"assets": [{**_RELEASE["assets"][0], "digest": "sha256:1111"}]}
    del release["assets"][0]["browser_download_url"]

    with pytest.raises(RuntimeError, match="browser_download_url"):
        _resolve("linux-x64", release=release)


_CID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
_OTHER_CID = "16fd2706-8baf-433b-82eb-8c7fada847da"


def _block(body: str) -> str:
    return f"<user_information>\n{body}\n</user_information>"


def _native_system(cid: str = _CID) -> ChatMessageSystem:
    return ChatMessageSystem(
        content="\n".join(
            [
                "<identity>You are a coding assistant.</identity>",
                _block(f"Conversation ID: {cid}"),
                "<rules>Be brief.</rules>",
            ]
        )
    )


def _primary(cid: str = _CID) -> list[ChatMessageSystem | ChatMessageUser]:
    return [_native_system(cid), ChatMessageUser(content="Do the task.")]


def _auxiliary() -> list[ChatMessageSystem | ChatMessageUser]:
    return [
        ChatMessageSystem(content="You are a conversation title generator"),
        ChatMessageUser(content="Do the task."),
    ]


def test_a_primary_request_yields_its_conversation_id() -> None:
    assert _native_conversation_id(_primary()) == _CID


def test_a_user_message_block_is_not_a_native_declaration() -> None:
    assert (
        _native_conversation_id(
            [
                ChatMessageSystem(content="You are a title generator"),
                ChatMessageUser(content=_block(f"Conversation ID: {_CID}")),
            ]
        )
        is None
    )


@pytest.mark.parametrize(
    "messages",
    [
        [_native_system("not-a-uuid")],
        [ChatMessageSystem(content=_block("Workspace: /repo"))],
        [
            ChatMessageSystem(
                content="\n".join(
                    [
                        _block(f"Conversation ID: {_CID}"),
                        _block(f"Conversation ID: {_OTHER_CID}"),
                    ]
                )
            )
        ],
        [
            ChatMessageSystem(
                content=(
                    "<user_information>\n"
                    f"Conversation ID: {_CID}\n"
                    "<rules>Be brief.</rules>"
                )
            )
        ],
        [ChatMessageSystem(content=f"Conversation ID: {_CID}\n</user_information>")],
    ],
)
def test_ambiguous_or_malformed_native_declarations_fail_loudly(
    messages: list[ChatMessageSystem],
) -> None:
    with pytest.raises(ValueError):
        _native_conversation_id(messages)


def test_unattended_accepts_only_its_first_native_conversation() -> None:
    conversation = _NativeConversation(unattended=True)

    assert conversation.accept(_auxiliary()) is False
    assert conversation.bound_id is None
    assert conversation.accept(_primary()) is True
    assert conversation.bound_id == _CID
    assert conversation.accept(_primary(_OTHER_CID)) is False


def test_unattended_reentry_binds_the_prior_conversation_before_requests() -> None:
    conversation = _NativeConversation(unattended=True, bound_id=_CID)

    assert conversation.accept(_primary(_OTHER_CID)) is False
    assert conversation.accept(_primary(_CID)) is True


def test_centaur_accepts_every_native_conversation_but_not_auxiliary_traffic() -> None:
    conversation = _NativeConversation(unattended=False)

    assert conversation.accept(_primary(_CID)) is True
    assert conversation.accept(_primary(_OTHER_CID)) is True
    assert conversation.accept(_auxiliary()) is False
