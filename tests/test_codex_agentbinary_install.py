import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, call, patch

import pytest
from inspect_ai.util import SandboxEnvironment
from inspect_swe._codex_cli import agentbinary as codex_agentbinary
from inspect_swe._util import agentbinary as install_agentbinary
from inspect_swe._util.agentbinary import (
    AgentBinarySource,
    ensure_agent_binary_installed,
)
from inspect_swe._util.sandbox import SANDBOX_INSTALL_DIR, SandboxPlatform

_VERSION = "0.147.0"
_PLATFORM: SandboxPlatform = "linux-x64"
_ARCH = "x86_64-unknown-linux-musl"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _release() -> str:
    return json.dumps(
        {
            "assets": [
                {
                    "name": f"codex-{_ARCH}.tar.gz",
                    "digest": f"sha256:{'a' * 64}",
                    "browser_download_url": "https://downloads.example/codex.tar.gz",
                },
                {
                    "name": f"codex-code-mode-host-{_ARCH}.tar.gz",
                    "digest": f"sha256:{'b' * 64}",
                    "browser_download_url": "https://downloads.example/host.tar.gz",
                },
            ]
        }
    )


def _source_with_cache(cache_dir: Path) -> AgentBinarySource:
    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        return cache_dir / f"codex-{version}-{platform}"

    source = codex_agentbinary.codex_cli_binary_source()
    source.cached_binary_path = cached_binary_path
    source.post_download = _extract_archive
    return source


def _sandbox() -> tuple[SandboxEnvironment, AsyncMock]:
    write_file = AsyncMock()
    sandbox = cast(
        SandboxEnvironment, cast(object, SimpleNamespace(write_file=write_file))
    )
    return sandbox, write_file


def _extract_archive(archive: bytes) -> bytes:
    return b"extracted-" + archive


@pytest.mark.anyio
async def test_codex_install_caches_code_mode_host(tmp_path: Path) -> None:
    # Given
    source = _source_with_cache(tmp_path)
    sandbox, _ = _sandbox()
    host_cache_path = tmp_path / f"codex-code-mode-host-{_VERSION}-{_PLATFORM}"

    with (
        patch.object(
            codex_agentbinary, "download_text_file", AsyncMock(return_value=_release())
        ),
        patch.object(
            codex_agentbinary, "extract_tarball", side_effect=_extract_archive
        ),
        patch.object(
            install_agentbinary,
            "download_file",
            AsyncMock(side_effect=[b"codex", b"host"]),
        ),
        patch.object(install_agentbinary, "verify_checksum", return_value=True),
        patch.object(
            install_agentbinary,
            "detect_sandbox_platform",
            AsyncMock(return_value=_PLATFORM),
        ),
        patch.object(install_agentbinary, "_resolved_versions", {}),
        patch.object(install_agentbinary, "trace"),
        patch.object(install_agentbinary, "sandbox_exec", AsyncMock()),
    ):
        # When
        await ensure_agent_binary_installed(source, version=_VERSION, sandbox=sandbox)

    # Then
    assert host_cache_path.read_bytes() == b"extracted-host"


@pytest.mark.anyio
async def test_codex_install_uses_warm_primary_and_host_caches_offline(
    tmp_path: Path,
) -> None:
    # Given
    source = _source_with_cache(tmp_path)
    primary_cache_path = source.cached_binary_path(_VERSION, _PLATFORM)
    host_cache_path = tmp_path / f"codex-code-mode-host-{_VERSION}-{_PLATFORM}"
    primary_cache_path.write_bytes(b"cached-codex")
    host_cache_path.write_bytes(b"cached-host")
    sandbox, write_file = _sandbox()

    with (
        patch.object(
            source,
            "resolve_version",
            AsyncMock(side_effect=AssertionError("must not resolve a warm cache")),
        ) as resolve_version,
        patch.object(
            install_agentbinary,
            "download_file",
            AsyncMock(side_effect=AssertionError("must not download a warm cache")),
        ) as download_file,
        patch.object(
            install_agentbinary,
            "detect_sandbox_platform",
            AsyncMock(return_value=_PLATFORM),
        ),
        patch.object(install_agentbinary, "trace"),
        patch.object(install_agentbinary, "sandbox_exec", AsyncMock()),
    ):
        # When
        installed = await ensure_agent_binary_installed(
            source, version=_VERSION, sandbox=sandbox
        )

    # Then
    assert installed == f"{SANDBOX_INSTALL_DIR}/codex-{_VERSION}-{_PLATFORM}"
    assert write_file.await_args_list == [
        call(installed, b"cached-codex"),
        call(f"{SANDBOX_INSTALL_DIR}/codex-code-mode-host", b"cached-host"),
    ]
    resolve_version.assert_not_awaited()
    download_file.assert_not_awaited()


@pytest.mark.anyio
async def test_codex_install_rejects_invalid_code_mode_host_checksum(
    tmp_path: Path,
) -> None:
    # Given
    source = _source_with_cache(tmp_path)
    sandbox, _ = _sandbox()

    with (
        patch.object(
            codex_agentbinary, "download_text_file", AsyncMock(return_value=_release())
        ),
        patch.object(
            codex_agentbinary, "extract_tarball", side_effect=_extract_archive
        ),
        patch.object(
            install_agentbinary,
            "download_file",
            AsyncMock(side_effect=[b"codex", b"host"]),
        ),
        patch.object(
            install_agentbinary,
            "verify_checksum",
            side_effect=[True, False],
        ),
        patch.object(
            install_agentbinary,
            "detect_sandbox_platform",
            AsyncMock(return_value=_PLATFORM),
        ),
        patch.object(install_agentbinary, "_resolved_versions", {}),
        patch.object(install_agentbinary, "trace"),
        patch.object(install_agentbinary, "sandbox_exec", AsyncMock()),
    ):
        # When / Then
        with pytest.raises(ValueError, match="Checksum verification failed"):
            await ensure_agent_binary_installed(
                source, version=_VERSION, sandbox=sandbox
            )
