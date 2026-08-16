import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal, NamedTuple

from inspect_ai.util import SandboxEnvironment, concurrency
from inspect_ai.util import sandbox as sandbox_env

from inspect_swe._util.trace import trace

from .checksum import verify_checksum
from .download import download_file
from .sandbox import (
    SANDBOX_INSTALL_DIR,
    SandboxPlatform,
    bash_command,
    detect_sandbox_platform,
    sandbox_exec,
)


class AgentBinaryVersion(NamedTuple):
    version: str
    expected_checksum: str
    download_url: str
    additional_binaries: tuple["AgentBinaryAdditional", ...] = ()


class AgentBinaryAdditional(NamedTuple):
    binary: str
    expected_checksum: str
    download_url: str
    post_download: Callable[[bytes], bytes] | None


@dataclass
class AgentBinarySource:
    agent: str
    binary: str
    resolve_version: Callable[
        [Literal["stable", "latest"] | str, SandboxPlatform],
        Awaitable[AgentBinaryVersion],
    ]
    cached_binary_path: Callable[[str, SandboxPlatform], Path]
    list_cached_binaries: Callable[[], list[Path]]
    post_download: Callable[[bytes], bytes] | None
    post_install: str | None
    additional_binary_names: tuple[str, ...] = ()


# In-process cache for version resolution results. When many samples run
# concurrently they all call resolve_version with the same arguments.
# Without caching, each call hits upstream APIs (e.g. GitHub, GCS),
# risking rate-limit exhaustion. We use a threading.Lock (not anyio.Lock)
# because it only guards synchronous dict reads/writes — never held across
# an await — and avoids issues with module-level anyio.Lock binding to a
# stale event loop across multiple anyio.run() calls. No expiry: entries
# live for the process lifetime. Two callers may race on the same key and
# both call resolve_version(), but this is benign (same result) and unlikely
# given the per-binary concurrency(1) lock in ensure_agent_binary_installed.
_resolve_version_lock = threading.Lock()
_resolved_versions: dict[tuple[str, str, SandboxPlatform], AgentBinaryVersion] = {}


async def ensure_agent_binary_installed(
    source: AgentBinarySource,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    user: str | None = None,
    sandbox: SandboxEnvironment | None = None,
) -> str:
    # resolve sandbox
    sandbox = sandbox or sandbox_env()

    # look in the sandbox first if we need to
    if version == "auto" or version == "sandbox":
        result = await sandbox.exec(bash_command(f"which {source.binary}"), user=user)
        if result.success:
            binary_path = result.stdout.strip()
            trace(f"Using {source.agent} installed in sandbox: {binary_path}")
            return binary_path

        # if version == "sandbox" and we don't find it that's an error
        if version == "sandbox":
            raise RuntimeError(f"unable to locate {source.agent} in sandbox")

        # otherwise set to "stable"
        version = "stable"

    # detect the sandbox target platform
    platform = await detect_sandbox_platform(sandbox)

    # use concurrency so multiple samples don't attempt the same download all at once
    async with concurrency(f"{source.binary}-install", 1, visible=False):
        additional_binary_data: tuple[tuple[str, bytes], ...] = ()
        if version not in ["stable", "latest"]:
            binary_bytes: bytes | None = read_cached_binary(
                source, version, platform, None
            )
            if binary_bytes is not None:
                primary_cache_path = source.cached_binary_path(version, platform)
                cached_additional_binary_data: list[tuple[str, bytes]] = []
                for additional_name in source.additional_binary_names:
                    additional_bytes = _read_cached_additional_binary(
                        _additional_binary_cache_path(
                            source, primary_cache_path, additional_name
                        )
                    )
                    if additional_bytes is None:
                        binary_bytes = None
                        break
                    cached_additional_binary_data.append(
                        (additional_name, additional_bytes)
                    )
                if binary_bytes is not None:
                    additional_binary_data = tuple(cached_additional_binary_data)
                    trace(
                        f"Used {source.agent} binary from cache: {version} ({platform})"
                    )
        else:
            binary_bytes = None

        if binary_bytes is None:
            binary_bytes, resolved = await download_agent_binary_async(
                source, version, platform, trace
            )
            resolved_version = resolved.version
            primary_cache_path = source.cached_binary_path(resolved_version, platform)
            downloaded_additional_binary_data: list[tuple[str, bytes]] = []
            for additional_binary in resolved.additional_binaries:
                downloaded_additional_binary_data.append(
                    (
                        additional_binary.binary,
                        await _download_additional_binary(
                            source, primary_cache_path, additional_binary
                        ),
                    )
                )
            additional_binary_data = tuple(downloaded_additional_binary_data)
        else:
            resolved_version = version

        # write it into the container and return it
        binary_path = (
            f"{SANDBOX_INSTALL_DIR}/{source.binary}-{resolved_version}-{platform}"
        )
        await sandbox.write_file(binary_path, binary_bytes)
        await sandbox_exec(sandbox, f"chmod +x {binary_path}", user="root")
        for additional_name, additional_bytes in additional_binary_data:
            additional_path = f"{SANDBOX_INSTALL_DIR}/{additional_name}"
            await sandbox.write_file(additional_path, additional_bytes)
            await sandbox_exec(sandbox, f"chmod +x {additional_path}", user="root")
        if source.post_install:
            await sandbox_exec(
                sandbox, f"{binary_path} {source.post_install}", user=user
            )
        return binary_path


async def _download_additional_binary(
    source: AgentBinarySource,
    primary_cache_path: Path,
    additional_binary: AgentBinaryAdditional,
) -> bytes:
    cache_path = _additional_binary_cache_path(
        source, primary_cache_path, additional_binary.binary
    )
    cached_binary = _read_cached_additional_binary(cache_path)
    if cached_binary is not None:
        return cached_binary

    binary_data = await download_file(additional_binary.download_url)
    if not verify_checksum(binary_data, additional_binary.expected_checksum):
        raise ValueError("Checksum verification failed")
    if additional_binary.post_download is not None:
        binary_data = additional_binary.post_download(binary_data)
    _write_cached_additional_binary(cache_path, binary_data)
    return binary_data


def _additional_binary_cache_path(
    source: AgentBinarySource, primary_cache_path: Path, binary: str
) -> Path:
    suffix = primary_cache_path.name.removeprefix(f"{source.binary}-")
    return primary_cache_path.with_name(f"{binary}-{suffix}")


def _read_cached_additional_binary(cache_path: Path) -> bytes | None:
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as cache_file:
        binary_data = cache_file.read()
    cache_path.touch()
    return binary_data


def _write_cached_additional_binary(cache_path: Path, binary_data: bytes) -> None:
    with open(cache_path, "wb") as cache_file:
        cache_file.write(binary_data)


async def download_agent_binary_async(
    source: AgentBinarySource,
    version: Literal["stable", "latest"] | str,
    platform: SandboxPlatform,
    logger: Callable[[str], None] | None = None,
) -> tuple[bytes, AgentBinaryVersion]:
    # resolve logger
    logger = logger or print

    # determine version and checksum (cached so concurrent samples don't
    # repeat upstream API calls that may be rate-limited)
    cache_key = (source.binary, version, platform)
    with _resolve_version_lock:
        cached = _resolved_versions.get(cache_key)
    if cached is not None:
        resolved = cached
    else:
        resolved = await source.resolve_version(version, platform)
        with _resolve_version_lock:
            _resolved_versions[cache_key] = resolved
    version = resolved.version
    expected_checksum = resolved.expected_checksum
    download_url = resolved.download_url

    # check the cache (if post_download is used, don't verify checksum since cached is processed)
    cache_checksum = None if source.post_download else expected_checksum
    binary_data = read_cached_binary(source, version, platform, cache_checksum)
    if binary_data is None:
        # not in cache, download and verify checksum
        binary_data = await download_file(download_url)
        if not verify_checksum(binary_data, expected_checksum):
            raise ValueError("Checksum verification failed")

        # apply post-download processing if provided (e.g., extract from tar.gz)
        if source.post_download is not None:
            binary_data = source.post_download(binary_data)

        # save to cache
        write_cached_binary(source, binary_data, version, platform)

        # trace
        logger(f"Downloaded {source.agent} binary: {version} ({platform})")
    else:
        logger(f"Used {source.agent} binary from cache: {version} ({platform})")

    # return data and resolved version
    return binary_data, resolved


def read_cached_binary(
    source: AgentBinarySource,
    version: str,
    platform: SandboxPlatform,
    expected_checksum: str | None,
) -> bytes | None:
    # no cached binary
    cache_path = source.cached_binary_path(version, platform)
    if not cache_path.exists():
        return None

    # read binary
    with open(cache_path, "rb") as f:
        binary_data = f.read()

    if expected_checksum is None or verify_checksum(binary_data, expected_checksum):
        cache_path.touch()
        return binary_data
    else:
        cache_path.unlink()
        return None


def write_cached_binary(
    source: AgentBinarySource,
    binary_data: bytes,
    version: str,
    platform: SandboxPlatform,
) -> None:
    binary_path = source.cached_binary_path(version, platform)

    with open(binary_path, "wb") as f:
        f.write(binary_data)

    _cleanup_binary_cache(source, keep_count=3)


def _cleanup_binary_cache(source: AgentBinarySource, keep_count: int = 5) -> None:
    # get all cached binaries
    cache_files = source.list_cached_binaries()
    if len(cache_files) <= keep_count:
        return

    # remove oldest
    cache_files.sort(key=lambda f: f.stat().st_atime)
    files_to_remove = cache_files[:-keep_count]
    for file_path in files_to_remove:
        file_path.unlink()
