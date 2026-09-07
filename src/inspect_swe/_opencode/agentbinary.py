import json
import re
import shlex
from pathlib import Path
from typing import Any

from inspect_ai.util import SandboxEnvironment, concurrency
from typing_extensions import Literal

from .._util.agentbinary import (
    AgentBinarySource,
    AgentBinaryVersion,
    ensure_agent_binary_installed,
)
from .._util.appdirs import package_cache_dir
from .._util.download import download_text_file
from .._util.node import create_npm_bundle, ensure_node_available
from .._util.ripgrep import ensure_ripgrep_available
from .._util.sandbox import (
    SANDBOX_INSTALL_DIR,
    SandboxPlatform,
    bash_command,
    detect_sandbox_platform,
)


async def ensure_opencode_setup(
    sandbox: SandboxEnvironment,
    version: Literal["auto", "sandbox", "stable", "latest"] | str,
    user: str | None,
) -> tuple[str, list[str]]:
    """Install OpenCode and return its binary plus dependency bin directories.

    OpenCode ships a standalone, self-contained binary per platform on GitHub
    releases, so it is acquired through the same host-side download-and-stage path
    as claude_code and codex_cli (``ensure_agent_binary_installed``): a ``which``
    probe reuses an already-installed/overlaid binary, otherwise the runner fetches
    the release tarball, verifies its checksum, and writes the binary into the
    sandbox. Nothing runs ``npm`` — and nothing downloads from inside the sandbox —
    so this works in a network-isolated sandbox where an in-sandbox fetch would hang.
    """
    platform = await detect_sandbox_platform(sandbox)

    node_binary = await ensure_node_available(sandbox, platform, user)
    dependency_bin_dirs = [
        node_binary.rsplit("/", 1)[0],
        await ensure_ripgrep_available(sandbox, platform, user),
    ]

    opencode_binary = await ensure_agent_binary_installed(
        opencode_binary_source(), version, user, sandbox
    )
    return opencode_binary, dependency_bin_dirs


def opencode_binary_source() -> AgentBinarySource:
    cached_binary_dir = package_cache_dir("opencode-downloads")

    async def resolve_version(
        version: Literal["stable", "latest"] | str, platform: SandboxPlatform
    ) -> AgentBinaryVersion:
        # Resolve the version and its release assets together:
        # /releases/latest returns both the tag name and the full asset list
        # in one response, so "stable"/"latest" resolves with exactly one
        # API call rather than a tag lookup followed by a second, per-tag
        # lookup. Halving the request count also halves how often a sample
        # queued behind a rate-limited neighbor has to repeat the exact call
        # that just failed. A pinned version still needs its own,
        # unavoidable per-tag lookup.
        if version in ["stable", "latest"]:
            release = await _fetch_latest_release()
            version = str(release["tag_name"]).lstrip("v")
        else:
            release = await _fetch_release_assets(version)

        assets = {a["name"]: a for a in release.get("assets", [])}
        asset = None
        for asset_name in _asset_name_candidates(platform):
            asset = assets.get(asset_name)
            if asset is not None:
                break
        if asset is None:
            raise RuntimeError(
                f"No matching asset for platform {platform!r} in opencode "
                f"release {version}"
            )

        # Extract checksum (format: "sha256:xxx"). GitHub serves
        # "digest": null for assets uploaded before digest support, so
        # normalize None to "" rather than crashing on .startswith.
        digest = asset.get("digest") or ""
        if not digest.startswith("sha256:"):
            raise RuntimeError(f"Invalid digest format: {digest}")
        expected_checksum = digest[7:]  # Remove "sha256:" prefix

        download_url = asset["browser_download_url"]
        # The release asset is a tar.gz wrapping the single opencode binary.
        # Cache it verbatim, checksum and all — as a "package" archive in
        # AgentBinarySource terms — and let ensure_agent_binary_installed
        # extract it in the sandbox at install time, rather than transforming
        # it host-side (post_download) into a blob a later cache hit can no
        # longer verify against the release digest.
        return AgentBinaryVersion(
            version, expected_checksum, download_url, package=True
        )

    def cached_binary_path(version: str, platform: SandboxPlatform) -> Path:
        # Never written or read in practice: resolve_version always returns
        # package=True, and opencode never shipped a single-binary cache (the
        # npm era cached bundles under "opencode-bundles"). Defined only
        # because AgentBinarySource requires it.
        return cached_binary_dir / f"opencode-{version}-{platform}"

    def cached_package_path(version: str, platform: SandboxPlatform) -> Path:
        return cached_binary_dir / f"opencode-package-{version}-{platform}.tar.gz"

    def list_cached_binaries() -> list[Path]:
        return list(cached_binary_dir.glob("opencode-*"))

    return AgentBinarySource(
        agent="opencode",
        binary="opencode",
        resolve_version=resolve_version,
        cached_binary_path=cached_binary_path,
        list_cached_binaries=list_cached_binaries,
        post_download=None,
        post_install=None,
        package_entrypoint="opencode",
        cached_package_path=cached_package_path,
    )


def _asset_name_candidates(platform: SandboxPlatform) -> list[str]:
    """Ordered release-asset name candidates for a platform, most preferred first.

    Every x64 release ships two builds: the default, compiled with AVX2
    instructions, and a "-baseline" build without them. OpenCode's own npm
    postinstall probes the *host's* CPU at install time and picks between
    them; we install host-side, before any in-sandbox probe ever runs, and
    can't reliably read the underlying host's CPU flags from out here — the
    sandbox may be scheduled onto a different physical host than the one
    that eventually runs the binary, and some sandbox providers don't expose
    real CPU flags at all. We deliberately prefer the baseline asset on
    every x64 platform: it trades a little performance on modern hosts for
    never staging a binary that SIGILLs on any host predating AVX2 (~2013),
    which is the safer default for sandbox portability. arm64 has no
    baseline variant (AVX2 is an x86 extension) and needs no fallback.
    """
    if platform in ("linux-x64", "linux-x64-musl"):
        musl = "-musl" if platform.endswith("-musl") else ""
        return [
            f"opencode-linux-x64-baseline{musl}.tar.gz",
            f"opencode-linux-x64{musl}.tar.gz",
        ]
    return [f"opencode-{platform}.tar.gz"]


async def _fetch_latest_release() -> dict[str, Any]:
    """Fetch the latest opencode release, tag and assets together."""
    latest_url = "https://api.github.com/repos/anomalyco/opencode/releases/latest"
    result: dict[str, Any] = json.loads(await download_text_file(latest_url))
    return result


async def _fetch_release_assets(version: str) -> dict[str, Any]:
    """Fetch release assets for a specific pinned version."""
    tag = f"v{version}"
    release_url = f"https://api.github.com/repos/anomalyco/opencode/releases/tags/{tag}"
    release_json = await download_text_file(release_url)
    result: dict[str, Any] = json.loads(release_json)
    return result


_CONFIG_DEPENDENCY_PACKAGE = "@opencode-ai/plugin"
_CONFIG_DEPENDENCY_CACHE_NAME = "opencode-config-deps"

_CONFIG_DEPENDENCY_VALIDATION = r"""
const fs = require("node:fs");
const path = require("node:path");
const [directory, expectedVersion, targetPlatform = `${process.platform}-${process.arch}`] = process.argv.slice(1);

function fail(message) {
  console.error(message);
  process.exit(1);
}

function readJson(relativePath) {
  const file = path.join(directory, relativePath);
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    fail(`unable to read ${file}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

const packageJson = readJson("package.json");
const packageLock = readJson("package-lock.json");
const root = packageLock.packages && packageLock.packages[""];
if (!root || !root.dependencies) fail("package-lock.json has no root production dependencies");
if (packageJson.dependencies?.["@opencode-ai/plugin"] !== expectedVersion) {
  fail(`package.json does not pin @opencode-ai/plugin@${expectedVersion}`);
}
if (root.dependencies["@opencode-ai/plugin"] !== expectedVersion) {
  fail(`package-lock.json does not pin @opencode-ai/plugin@${expectedVersion}`);
}
const plugin = readJson("node_modules/@opencode-ai/plugin/package.json");
if (plugin.version !== expectedVersion) {
  fail(`installed @opencode-ai/plugin is ${plugin.version}, expected ${expectedVersion}`);
}
const [targetOs, targetCpu] = targetPlatform.split("-");
if (!targetOs || !targetCpu) fail(`invalid target platform ${targetPlatform}`);

function selectorIncludesTarget(selectors, target) {
  if (!Array.isArray(selectors)) return true;
  const values = selectors.filter((value) => typeof value === "string");
  const positives = values.filter((value) => !value.startsWith("!"));
  return !values.includes(`!${target}`) &&
    (positives.length === 0 || positives.includes(target));
}

function appliesToTarget(record) {
  return selectorIncludesTarget(record.os, targetOs) &&
    selectorIncludesTarget(record.cpu, targetCpu);
}

const packagePaths = Object.keys(packageLock.packages || {}).filter((entry) =>
  entry.startsWith("node_modules/"),
);
if (packagePaths.length === 0) fail("package-lock.json has no installed production closure");
for (const packagePath of packagePaths) {
  const expected = packageLock.packages[packagePath];
  if (!expected || typeof expected.version !== "string") {
    fail(`package-lock.json has no version for ${packagePath}`);
  }
  const installedPackage = path.join(directory, packagePath, "package.json");
  if (!fs.existsSync(installedPackage)) {
    if (expected.optional === true && !appliesToTarget(expected)) continue;
    fail(`missing installed package metadata for ${packagePath}`);
  }
  if (!appliesToTarget(expected)) continue;
  const installed = readJson(`${packagePath}/package.json`);
  if (installed.version !== expected.version) {
    fail(`installed ${packagePath} is ${installed.version}, expected ${expected.version}`);
  }
}
"""


async def seed_opencode_config_dependencies(
    sandbox: SandboxEnvironment,
    opencode_binary: str,
    node_binary: str,
    dependency_bin_dirs: list[str],
    config_dirs: list[str],
    config_dependency_seed: str | None,
    user: str | None,
) -> None:
    """Seed every native OpenCode config directory before CLI initialization."""
    cli_version = await _opencode_cli_version(
        sandbox, opencode_binary, dependency_bin_dirs, user
    )
    if config_dependency_seed is not None:
        if not config_dependency_seed.startswith("/"):
            raise ValueError(
                "config_dependency_seed must be an absolute sandbox directory"
            )
        seed_dir = config_dependency_seed
    else:
        platform = await detect_sandbox_platform(sandbox)
        seed_dir = await _local_config_dependency_seed(
            sandbox, node_binary, cli_version, platform, user
        )

    await _validate_config_dependency_tree(
        sandbox, node_binary, seed_dir, cli_version, user
    )
    for config_dir in dict.fromkeys(config_dirs):
        if not await _config_dependency_writable(sandbox, config_dir, user):
            continue

        state = await _config_dependency_state(sandbox, config_dir, user)
        if state == "partial":
            if not await _hook_owned_config_preparation(
                sandbox, config_dir, cli_version, user
            ):
                raise RuntimeError(
                    "OpenCode config directory "
                    f"{config_dir!r} has incomplete dependency metadata; refusing "
                    "to overwrite it"
                )
            await _recover_hook_owned_config_preparation(
                sandbox, config_dir, cli_version, user
            )
            state = await _config_dependency_state(sandbox, config_dir, user)
            if state != "empty":
                raise RuntimeError(
                    "Unable to recover interrupted OpenCode config dependency "
                    f"preparation in {config_dir!r}"
                )

        if state == "complete":
            await _validate_config_dependency_tree(
                sandbox, node_binary, config_dir, cli_version, user
            )
            continue

        await _stage_config_dependency_tree(
            sandbox, node_binary, seed_dir, config_dir, cli_version, user
        )


async def _opencode_cli_version(
    sandbox: SandboxEnvironment,
    opencode_binary: str,
    dependency_bin_dirs: list[str],
    user: str | None,
) -> str:
    path = ":".join([*dependency_bin_dirs, "/usr/local/bin", "/usr/bin", "/bin"])
    result = await sandbox.exec(
        [opencode_binary, "--version"], env={"PATH": path}, user=user
    )
    if not result.success:
        raise RuntimeError(
            "Unable to determine the OpenCode CLI version before provisioning "
            f"its config dependencies: {result.stderr}"
        )
    match = re.fullmatch(r"v?(\d+\.\d+\.\d+)", result.stdout.strip())
    if match is None:
        raise RuntimeError(
            "Unable to parse OpenCode CLI version before provisioning config "
            f"dependencies: {result.stdout.strip()!r}"
        )
    return match.group(1)


async def _local_config_dependency_seed(
    sandbox: SandboxEnvironment,
    node_binary: str,
    cli_version: str,
    platform: SandboxPlatform,
    user: str | None,
) -> str:
    seed_dir = f"{SANDBOX_INSTALL_DIR}/opencode-config-deps-{cli_version}-{platform}"
    staging_dir = f"{seed_dir}.staging"
    archive_path = f"{seed_dir}.tar.gz"
    async with concurrency(
        f"opencode-config-deps-{cli_version}-{platform}", 1, visible=False
    ):
        bundle_data = create_npm_bundle(
            package=_CONFIG_DEPENDENCY_PACKAGE,
            version=cli_version,
            platform=platform,
            cache_name=_CONFIG_DEPENDENCY_CACHE_NAME,
            ignore_scripts=True,
        )
        await sandbox.write_file(archive_path, bundle_data)
        result = await sandbox.exec(
            bash_command(
                f"rm -rf -- {shlex.quote(staging_dir)} && "
                f"mkdir -p {shlex.quote(staging_dir)} && "
                f"tar -xzf {shlex.quote(archive_path)} -C {shlex.quote(staging_dir)}"
            ),
            user="root",
        )
        if not result.success:
            raise RuntimeError(
                "Unable to extract the local OpenCode config dependency seed: "
                f"{result.stderr}"
            )
        await _validate_config_dependency_tree(
            sandbox, node_binary, staging_dir, cli_version, "root"
        )
        result = await sandbox.exec(
            bash_command(
                f"rm -rf -- {shlex.quote(seed_dir)} && "
                f"mv {shlex.quote(staging_dir)} {shlex.quote(seed_dir)} && "
                f"rm -f {shlex.quote(archive_path)}"
            ),
            user="root",
        )
        if not result.success:
            raise RuntimeError(
                "Unable to promote the local OpenCode config dependency seed: "
                f"{result.stderr}"
            )
    return seed_dir


def _config_dependency_staging_dir(config_dir: str, cli_version: str) -> str:
    return f"{config_dir}.inspect-swe-opencode-deps-{cli_version}.staging"


def _config_dependency_preparation_contents(cli_version: str) -> str:
    return f"inspect-swe-opencode-deps:{cli_version}"


def _config_dependency_preparation_marker(config_dir: str, cli_version: str) -> str:
    return f"{config_dir}/.inspect-swe-opencode-deps-{cli_version}.preparing"


async def _config_dependency_writable(
    sandbox: SandboxEnvironment, config_dir: str, user: str | None
) -> bool:
    quoted_dir = shlex.quote(config_dir)
    result = await sandbox.exec(
        bash_command(
            "# opencode-config-dependency-writable\n"
            f"target={quoted_dir}; "
            'if [ -e "$target" ]; then '
            'if [ -d "$target" ] && [ -w "$target" ]; then echo writable; '
            "else echo read-only; fi; "
            "else "
            'parent="$(dirname "$target")"; '
            'while [ ! -e "$parent" ]; do '
            'next="$(dirname "$parent")"; '
            '[ "$next" = "$parent" ] && break; '
            'parent="$next"; '
            "done; "
            'if [ -d "$parent" ] && [ -w "$parent" ]; then echo writable; '
            "else echo read-only; fi; "
            "fi"
        ),
        user=user,
    )
    state = result.stdout.strip()
    if not result.success or state not in {"writable", "read-only"}:
        raise RuntimeError(
            f"Unable to inspect OpenCode config directory permissions in {config_dir!r}: "
            f"{result.stderr}"
        )
    return state == "writable"


async def _hook_owned_config_preparation(
    sandbox: SandboxEnvironment,
    config_dir: str,
    cli_version: str,
    user: str | None,
) -> bool:
    marker = shlex.quote(_config_dependency_preparation_marker(config_dir, cli_version))
    expected_contents = shlex.quote(
        _config_dependency_preparation_contents(cli_version)
    )
    result = await sandbox.exec(
        bash_command(
            "# opencode-config-dependency-preparation\n"
            f"[ -f {marker} ] && "
            f'[ "$(cat {marker})" = {expected_contents} ] && '
            "echo present || echo absent"
        ),
        user=user,
    )
    state = result.stdout.strip()
    if not result.success or state not in {"present", "absent"}:
        raise RuntimeError(
            "Unable to inspect interrupted OpenCode config dependency preparation "
            f"in {config_dir!r}: {result.stderr}"
        )
    return state == "present"


async def _recover_hook_owned_config_preparation(
    sandbox: SandboxEnvironment,
    config_dir: str,
    cli_version: str,
    user: str | None,
) -> None:
    target = shlex.quote(config_dir)
    marker = shlex.quote(_config_dependency_preparation_marker(config_dir, cli_version))
    staging = shlex.quote(_config_dependency_staging_dir(config_dir, cli_version))
    result = await sandbox.exec(
        bash_command(
            f"rm -f -- {target}/package.json {target}/package-lock.json {marker} && "
            f"rm -rf -- {target}/node_modules {staging}"
        ),
        user=user,
    )
    if not result.success:
        raise RuntimeError(
            "Unable to recover interrupted OpenCode config dependency preparation "
            f"in {config_dir!r}: {result.stderr}"
        )


async def _stage_config_dependency_tree(
    sandbox: SandboxEnvironment,
    node_binary: str,
    seed_dir: str,
    config_dir: str,
    cli_version: str,
    user: str | None,
) -> None:
    source = shlex.quote(seed_dir)
    target = shlex.quote(config_dir)
    staging_dir = _config_dependency_staging_dir(config_dir, cli_version)
    staging = shlex.quote(staging_dir)
    marker = shlex.quote(_config_dependency_preparation_marker(config_dir, cli_version))
    marker_contents = shlex.quote(_config_dependency_preparation_contents(cli_version))
    result = await sandbox.exec(
        bash_command(
            f"rm -rf -- {staging} && "
            f"mkdir -p {target} {staging} && "
            f"cp -a --no-preserve=ownership "
            f"{source}/package.json "
            f"{source}/package-lock.json "
            f"{source}/node_modules {staging}/"
        ),
        user=user,
    )
    if not result.success:
        raise RuntimeError(
            f"Unable to stage OpenCode config dependencies in {config_dir!r}: "
            f"{result.stderr}"
        )
    await _validate_config_dependency_tree(
        sandbox, node_binary, staging_dir, cli_version, user
    )
    result = await sandbox.exec(
        bash_command(
            f"printf '%s\\n' {marker_contents} > {marker} && "
            f"mv {staging}/package.json {target}/package.json && "
            f"mv {staging}/package-lock.json {target}/package-lock.json && "
            f"mv {staging}/node_modules {target}/node_modules && "
            f"rmdir {staging} && "
            f"rm -f -- {marker}"
        ),
        user=user,
    )
    if not result.success:
        raise RuntimeError(
            f"Unable to promote OpenCode config dependencies in {config_dir!r}: "
            f"{result.stderr}"
        )
    await _validate_config_dependency_tree(
        sandbox, node_binary, config_dir, cli_version, user
    )


async def _config_dependency_state(
    sandbox: SandboxEnvironment, config_dir: str, user: str | None
) -> Literal["empty", "complete", "partial"]:
    quoted_dir = shlex.quote(config_dir)
    result = await sandbox.exec(
        bash_command(
            f"mkdir -p {quoted_dir}; "
            f"metadata=0; "
            f"[ -e {quoted_dir}/package.json ] && metadata=$((metadata + 1)); "
            f"[ -e {quoted_dir}/package-lock.json ] && metadata=$((metadata + 1)); "
            f"[ -e {quoted_dir}/node_modules ] && metadata=$((metadata + 1)); "
            f'if [ "$metadata" -eq 0 ]; then echo empty; '
            f'elif [ "$metadata" -eq 3 ]; then echo complete; '
            f"else echo partial; fi"
        ),
        user=user,
    )
    state = result.stdout.strip()
    if not result.success or state not in {"empty", "complete", "partial"}:
        raise RuntimeError(
            f"Unable to inspect OpenCode dependency metadata in {config_dir!r}: "
            f"{result.stderr}"
        )
    if state == "empty":
        return "empty"
    if state == "complete":
        return "complete"
    return "partial"


async def _validate_config_dependency_tree(
    sandbox: SandboxEnvironment,
    node_binary: str,
    directory: str,
    cli_version: str,
    user: str | None,
) -> None:
    result = await sandbox.exec(
        [node_binary, "-e", _CONFIG_DEPENDENCY_VALIDATION, directory, cli_version],
        user=user,
    )
    if not result.success:
        raise RuntimeError(
            "OpenCode config dependency seed is incompatible with the running "
            f"CLI {cli_version} at {directory!r}: {result.stderr.strip()}"
        )
