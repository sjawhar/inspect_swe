"""Focused Gemini CLI provisioning tests."""

import tarfile
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import anyio
import pytest
from inspect_swe._gemini_cli import agentbinary
from inspect_swe.gemini_cli_instrumentation import (
    GEMINI_CLI_INSTRUMENTED_VERSION,
    GEMINI_CLI_TRACE_CONTEXT_CACHE_REVISION,
    GeminiCliInstrumentationError,
)


def _bundle_with_marker(marker: str) -> bytes:
    buffer = BytesIO()
    marker_bytes = marker.encode()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("node_modules/instrumentation-target.txt")
        member.size = len(marker_bytes)
        archive.addfile(member, BytesIO(marker_bytes))
    return buffer.getvalue()


def _marker_from_bundle(bundle_data: bytes) -> str:
    with tarfile.open(fileobj=BytesIO(bundle_data), mode="r:gz") as archive:
        member = archive.extractfile("node_modules/instrumentation-target.txt")
        assert member is not None
        return member.read().decode()


def _instrumented_cache_path(cache_dir: Path) -> Path:
    return cache_dir / (
        "gemini-cli-instrumented-"
        f"{GEMINI_CLI_TRACE_CONTEXT_CACHE_REVISION}-"
        f"{GEMINI_CLI_INSTRUMENTED_VERSION}-linux-x64.tar.gz"
    )


def test_stages_and_patches_a_stock_gemini_bundle_before_caching_and_installing(
    tmp_path: Path,
) -> None:
    sandbox = Mock()
    sandbox.exec = AsyncMock(return_value=Mock(success=False))
    stock_bundle = _bundle_with_marker("stock")

    def patch_staged_tree(tree: Path, version: str) -> None:
        assert version == "0.58.0"
        (tree / "node_modules/instrumentation-target.txt").write_text("instrumented")

    with (
        patch.object(
            agentbinary,
            "create_npm_bundle",
            return_value=stock_bundle,
        ) as create_bundle,
        patch.object(
            agentbinary,
            "package_cache_dir",
            return_value=tmp_path,
            create=True,
        ),
        patch.object(
            agentbinary,
            "patch_gemini_cli_tree",
            side_effect=patch_staged_tree,
            create=True,
        ) as patch_tree,
        patch.object(
            agentbinary,
            "_verify_instrumented_gemini_bundle",
            create=True,
        ) as verify_bundle,
        patch.object(
            agentbinary,
            "install_npm_bundle",
            AsyncMock(return_value="/opt/inspect/bin/gemini"),
        ) as install_bundle,
    ):
        binary = anyio.run(
            agentbinary.ensure_gemini_cli_installed,
            sandbox,
            "/node",
            "0.58.0",
            "linux-x64",
        )

    assert binary == "/opt/inspect/bin/gemini"
    create_bundle.assert_called_once()
    patch_tree.assert_called_once()
    install_await = install_bundle.await_args
    assert install_await is not None
    instrumented_bundle: bytes = install_await.kwargs["bundle_data"]
    assert _marker_from_bundle(instrumented_bundle) == "instrumented"
    verify_bundle.assert_called_once_with(instrumented_bundle)
    assert (
        _marker_from_bundle(_instrumented_cache_path(tmp_path).read_bytes())
        == "instrumented"
    )


def test_installs_a_warm_instrumented_gemini_cache_without_repatching(
    tmp_path: Path,
) -> None:
    sandbox = Mock()
    sandbox.exec = AsyncMock(return_value=Mock(success=False))
    cache_path = _instrumented_cache_path(tmp_path)
    cache_path.write_bytes(_bundle_with_marker("instrumented"))

    with (
        patch.object(
            agentbinary,
            "create_npm_bundle",
            side_effect=AssertionError("warm instrumented cache must not download"),
        ),
        patch.object(
            agentbinary,
            "package_cache_dir",
            return_value=tmp_path,
            create=True,
        ),
        patch.object(
            agentbinary,
            "patch_gemini_cli_tree",
            side_effect=AssertionError("warm instrumented cache must not patch"),
            create=True,
        ),
        patch.object(
            agentbinary,
            "_verify_instrumented_gemini_bundle",
            create=True,
        ) as verify_bundle,
        patch.object(
            agentbinary,
            "install_npm_bundle",
            AsyncMock(return_value="/opt/inspect/bin/gemini"),
        ) as install_bundle,
    ):
        binary = anyio.run(
            agentbinary.ensure_gemini_cli_installed,
            sandbox,
            "/node",
            "0.58.0",
            "linux-x64",
        )
    assert binary == "/opt/inspect/bin/gemini"
    warm_install_await = install_bundle.await_args
    assert warm_install_await is not None
    warm_bundle: bytes = warm_install_await.kwargs["bundle_data"]
    assert _marker_from_bundle(warm_bundle) == "instrumented"
    verify_bundle.assert_called_once_with(cache_path.read_bytes())


def test_rejects_a_stock_bundle_in_the_instrumented_cache(tmp_path: Path) -> None:
    sandbox = Mock()
    sandbox.exec = AsyncMock(return_value=Mock(success=False))
    cache_path = _instrumented_cache_path(tmp_path)
    cache_path.write_bytes(_bundle_with_marker("stock"))

    with (
        patch.object(
            agentbinary,
            "create_npm_bundle",
            side_effect=AssertionError("invalid instrumented cache must not download"),
        ),
        patch.object(
            agentbinary,
            "package_cache_dir",
            return_value=tmp_path,
            create=True,
        ),
        patch.object(
            agentbinary,
            "patch_gemini_cli_tree",
            side_effect=AssertionError("invalid instrumented cache must not patch"),
            create=True,
        ),
        patch.object(
            agentbinary,
            "_verify_instrumented_gemini_bundle",
            side_effect=GeminiCliInstrumentationError(
                "invalid instrumented cache artifact"
            ),
            create=True,
        ),
        pytest.raises(
            GeminiCliInstrumentationError, match="invalid instrumented cache artifact"
        ),
    ):
        anyio.run(
            agentbinary.ensure_gemini_cli_installed,
            sandbox,
            "/node",
            "0.58.0",
            "linux-x64",
        )


@pytest.mark.parametrize("alias", ("auto", "stable", "latest"))
def test_instrumented_version_aliases_resolve_to_the_supported_bundle(
    alias: str,
) -> None:
    assert (
        anyio.run(agentbinary.resolve_gemini_version, alias)
        == GEMINI_CLI_INSTRUMENTED_VERSION
    )


def test_rejects_an_uninstrumented_gemini_version_before_downloading() -> None:
    sandbox = Mock()
    sandbox.exec = AsyncMock(return_value=Mock(success=False))

    with (
        patch.object(
            agentbinary,
            "create_npm_bundle",
            side_effect=AssertionError("unsupported version must not download"),
        ),
        pytest.raises(
            GeminiCliInstrumentationError,
            match="instrumentation supports only",
        ),
    ):
        anyio.run(
            agentbinary.ensure_gemini_cli_installed,
            sandbox,
            "/node",
            "0.57.0",
            "linux-x64",
        )


def test_sandbox_version_uses_the_attached_binary_without_network_resolution() -> None:
    sandbox = Mock()
    with (
        patch.object(
            agentbinary, "detect_sandbox_platform", AsyncMock(return_value="linux-x64")
        ),
        patch.object(
            agentbinary, "ensure_node_available", AsyncMock(return_value="/node")
        ),
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
        binary, node = anyio.run(
            agentbinary.ensure_gemini_cli_setup, sandbox, "sandbox", "agent"
        )

    assert (binary, node) == ("/usr/local/bin/gemini", "/node")
    installed_binary.assert_awaited_once_with(sandbox, "/node", "agent")


def test_sandbox_version_fails_loudly_when_the_attached_binary_is_missing() -> None:
    sandbox = Mock()
    with (
        patch.object(
            agentbinary, "detect_sandbox_platform", AsyncMock(return_value="linux-x64")
        ),
        patch.object(
            agentbinary, "ensure_node_available", AsyncMock(return_value="/node")
        ),
        patch.object(
            agentbinary,
            "_sandbox_gemini_binary",
            AsyncMock(
                side_effect=RuntimeError("attached Gemini CLI binary is unavailable")
            ),
        ),
        patch.object(
            agentbinary,
            "resolve_gemini_version",
            AsyncMock(side_effect=AssertionError("sandbox must not resolve latest")),
        ),
        pytest.raises(RuntimeError, match="attached Gemini CLI binary is unavailable"),
    ):
        anyio.run(agentbinary.ensure_gemini_cli_setup, sandbox, "sandbox", "agent")
