import json
import logging
import re
from pathlib import Path
from typing import Any

from inspect_ai.util import SandboxEnvironment, concurrency
from typing_extensions import Literal

from .._util.agentbinary import (
    AgentBinaryAdditional,
    AgentBinarySource,
    AgentBinaryVersion,
)
from .._util.appdirs import package_cache_dir
from .._util.download import download_text_file
from .._util.sandbox import SandboxPlatform, sandbox_exec
from .._util.tarball import extract_tarball
from ._bundled_catalog import BUNDLED_CODEX_CATALOG

logger = logging.getLogger(__name__)


def codex_cli_binary_source() -> AgentBinarySource:
    cached_binary_dir = package_cache_dir("codex-cli-downloads")

    async def resolve_version(
        version: Literal["stable", "latest"] | str, platform: SandboxPlatform
    ) -> AgentBinaryVersion:
        # Resolve version alias if needed
        if version in ["stable", "latest"]:
            version = await _fetch_latest_stable_version()

        # Get release information
        release = await _fetch_release_assets(version)

        arch = _platform_to_codex_arch(platform)
        codex_asset = _release_asset(release, f"codex-{arch}.tar.gz", platform)
        host_asset = _release_asset(
            release, f"codex-code-mode-host-{arch}.tar.gz", platform
        )

        return AgentBinaryVersion(
            version,
            _asset_checksum(codex_asset),
            codex_asset["browser_download_url"],
            (
                AgentBinaryAdditional(
                    "codex-code-mode-host",
                    _asset_checksum(host_asset),
                    host_asset["browser_download_url"],
                    extract_tarball,
                ),
            ),
        )

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_binary_dir / f"codex-{version}-{platform}"

    def list_cached_binaries() -> list[Path]:
        return list(cached_binary_dir.glob("codex-[0-9]*"))

    return AgentBinarySource(
        agent="codex cli",
        binary="codex",
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=list_cached_binaries,
        post_download=extract_tarball,
        post_install=None,
        additional_binary_names=("codex-code-mode-host",),
    )


def _platform_to_codex_arch(platform: SandboxPlatform) -> str:
    """Map SandboxPlatform to Codex architecture string.

    Always use musl variants for better compatibility since they're
    statically linked and don't depend on system GLIBC version.
    """
    platform_map = {
        "linux-x64": "x86_64-unknown-linux-musl",
        "linux-x64-musl": "x86_64-unknown-linux-musl",
        "linux-arm64": "aarch64-unknown-linux-musl",
        "linux-arm64-musl": "aarch64-unknown-linux-musl",
    }
    if platform not in platform_map:
        raise ValueError(f"Unsupported platform: {platform}")
    return platform_map[platform]


def _release_asset(
    release: dict[str, Any], asset_name: str, platform: SandboxPlatform
) -> dict[str, str]:
    for asset in release.get("assets", []):
        if not isinstance(asset, dict):
            continue
        name = asset.get("name")
        digest = asset.get("digest")
        download_url = asset.get("browser_download_url")
        if name == asset_name:
            if not isinstance(digest, str):
                raise RuntimeError(f"Invalid digest format: {digest}")
            if not isinstance(download_url, str):
                raise RuntimeError(f"Invalid download URL: {download_url}")
            return {"digest": digest, "browser_download_url": download_url}
    raise RuntimeError(f"No asset named {asset_name} found for platform {platform}")


def _asset_checksum(asset: dict[str, str]) -> str:
    digest = asset["digest"]
    if not digest.startswith("sha256:"):
        raise RuntimeError(f"Invalid digest format: {digest}")
    return digest[7:]


async def _fetch_latest_stable_version() -> str:
    """Fetch the latest stable version from GitHub releases."""
    # Use the single-release `latest` endpoint (excludes prereleases/drafts by
    # definition). The full releases listing is so large for openai/codex that
    # GitHub's API frequently 504s generating it.
    latest_url = "https://api.github.com/repos/openai/codex/releases/latest"
    latest = json.loads(await download_text_file(latest_url))
    tag_name = latest["tag_name"]

    # Extract version from tag (e.g., "rust-v0.29.0" -> "0.29.0")
    if tag_name.startswith("rust-v"):
        result: str = tag_name[6:]  # Remove "rust-v" prefix
        return result
    else:
        raise RuntimeError(f"Unexpected tag format: {tag_name}")


async def _fetch_release_assets(version: str) -> dict[str, Any]:
    """Fetch release assets for a specific version."""
    tag = f"rust-v{version}"
    release_url = f"https://api.github.com/repos/openai/codex/releases/tags/{tag}"
    release_json = await download_text_file(release_url)
    result: dict[str, Any] = json.loads(release_json)
    return result


async def codex_binary_version(
    sandbox: SandboxEnvironment, binary: str, user: str | None = None
) -> str | None:
    """Resolve the concrete version of an installed codex binary.

    Returns a semver string (e.g. ``"0.50.0"``) parsed from ``codex --version``,
    or ``None`` if it can't be determined.
    """
    try:
        output = await sandbox_exec(sandbox, f"{binary} --version", user=user)
    except RuntimeError:
        return None
    match = re.search(r"(\d+\.\d+\.\d+)", output)
    return match.group(1) if match else None


def _cached_catalog_path(version: str) -> Path:
    return package_cache_dir("codex-cli-downloads") / f"codex-{version}-models.json"


async def codex_models_catalog(version: str | None) -> dict[str, Any]:
    """Resolve the native Codex model catalog for a specific version.

    Prefers the version-matched ``models.json`` from the corresponding
    ``rust-v{version}`` release tag, cached alongside the binary so the offline
    guarantee matches the binary's. When the version is unknown or the fetch
    fails (offline, rate-limited, or a pre-``models-manager`` release), falls back
    to the bundled snapshot (``BUNDLED_CODEX_CATALOG``) so model alignment stays
    deterministic rather than degrading to Codex's generic fallback.

    The fetch is serialized with a single-slot concurrency lock so parallel
    samples don't stampede ``raw.githubusercontent.com`` (the first writes the
    cache; the rest read it).
    """
    cached = _read_cached_catalog(version)
    if cached is not None:
        return cached

    if version:
        async with concurrency("codex-models-catalog", 1, visible=False):
            # re-check the cache now that we hold the lock (another sample may
            # have fetched and written it while we were waiting).
            cached = _read_cached_catalog(version)
            if cached is not None:
                return cached
            fetched = await _fetch_models_catalog(version)
            if fetched is not None:
                return fetched

    return BUNDLED_CODEX_CATALOG


def _read_cached_catalog(version: str | None) -> dict[str, Any] | None:
    """Return the cached catalog for ``version`` if present and parseable."""
    if not version:
        return None
    cache_path = _cached_catalog_path(version)
    if not cache_path.exists():
        return None
    try:
        return cast_catalog(json.loads(cache_path.read_text()))
    except (OSError, json.JSONDecodeError):
        return None


async def _fetch_models_catalog(version: str) -> dict[str, Any] | None:
    """Fetch and cache the version-matched catalog, or ``None`` on failure."""
    url = (
        "https://raw.githubusercontent.com/openai/codex/"
        f"rust-v{version}/codex-rs/models-manager/models.json"
    )
    try:
        text = await download_text_file(url)
        catalog = cast_catalog(json.loads(text))
    except Exception as ex:
        logger.debug(f"Unable to fetch codex model catalog for {version}: {ex}")
        return None

    if catalog is None:
        return None

    try:
        _cached_catalog_path(version).write_text(text)
    except OSError:
        pass
    return catalog


def cast_catalog(data: Any) -> dict[str, Any] | None:
    """Return ``data`` if it is a catalog-shaped dict, else ``None``."""
    return data if isinstance(data, dict) else None
