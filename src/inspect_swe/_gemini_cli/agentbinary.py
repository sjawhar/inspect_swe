import hashlib
import shutil
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Literal

from inspect_ai.util import SandboxEnvironment, concurrency

from .._util.appdirs import package_cache_dir
from .._util.node import (
    create_npm_bundle,
    ensure_node_available,
    install_npm_bundle,
)
from .._util.sandbox import (
    SANDBOX_INSTALL_DIR,
    SandboxPlatform,
    bash_command,
    detect_sandbox_platform,
)
from ..gemini_cli_instrumentation import (
    GEMINI_CLI_INSTRUMENTED_VERSION,
    GEMINI_CLI_TRACE_CONTEXT_CACHE_REVISION,
    GEMINI_CLI_TRACE_CONTEXT_CONTRACT,
    GeminiCliInstrumentationError,
    patch_gemini_cli_tree,
)


async def ensure_gemini_cli_setup(
    sandbox: SandboxEnvironment,
    version: Literal["auto", "sandbox", "stable", "latest"] | str,
    user: str | None,
) -> tuple[str, str]:
    """Install node and Gemini CLI in the sandbox.

    Returns the Gemini executable and Node.js binary paths.
    """
    platform = await detect_sandbox_platform(sandbox)
    node_binary = await ensure_node_available(sandbox, platform, user)
    if version == "sandbox":
        return await _sandbox_gemini_binary(sandbox, node_binary, user), node_binary

    gemini_version = await resolve_gemini_version(version)
    gemini_binary = await ensure_gemini_cli_installed(
        sandbox, node_binary, gemini_version, platform, user
    )
    return gemini_binary, node_binary


async def resolve_gemini_version(
    version: Literal["auto", "sandbox", "stable", "latest"] | str,
) -> str:
    """Resolve aliases to the Gemini CLI version with native tracing support."""
    if version in {"auto", "stable", "latest"}:
        return GEMINI_CLI_INSTRUMENTED_VERSION
    return version


async def _sandbox_gemini_binary(
    sandbox: SandboxEnvironment, node_path: str, user: str | None
) -> str:
    """Return the pre-attached Gemini executable without consulting a release channel."""
    result = await sandbox.exec(bash_command("which gemini"), user=user)
    if not result.success or not result.stdout.strip():
        raise RuntimeError(
            "Gemini CLI version='sandbox' requires an attached gemini executable"
        )
    gemini_binary = result.stdout.strip()
    version_result = await sandbox.exec(
        cmd=[node_path, gemini_binary, "--version"], user=user
    )
    if not version_result.success:
        raise RuntimeError(
            "attached Gemini CLI executable failed its version check:\n"
            f"stdout: {version_result.stdout}\n"
            f"stderr: {version_result.stderr}"
        )
    return gemini_binary


async def ensure_gemini_cli_installed(
    sandbox: SandboxEnvironment,
    node_path: str,
    version: str,
    platform: SandboxPlatform,
    user: str | None = None,
) -> str:
    """Install Gemini CLI via npm and return path to the gemini binary.

    This installs the full @google/gemini-cli package including all policy files,
    ensuring YOLO mode and other features work correctly.
    """
    gemini_install_dir = f"{SANDBOX_INSTALL_DIR}/gemini-cli"
    gemini_binary = f"{gemini_install_dir}/node_modules/.bin/gemini"
    if version != GEMINI_CLI_INSTRUMENTED_VERSION:
        raise GeminiCliInstrumentationError(
            "Gemini W3C trace-context instrumentation supports only "
            f"{GEMINI_CLI_INSTRUMENTED_VERSION}, not {version}"
        )

    result = await sandbox.exec(
        bash_command(f"test -x {gemini_binary}"),
        user=user,
    )
    if result.success:
        result = await sandbox.exec(
            cmd=[node_path, gemini_binary, "--version"],
            user=user,
        )
        if result.success:
            installed_version = result.stdout.strip()
            if installed_version == version:
                return gemini_binary

    async with concurrency("gemini-cli-install", 1, visible=False):
        bundle_data = _instrumented_gemini_bundle(version, platform)
        return await install_npm_bundle(
            sandbox=sandbox,
            bundle_data=bundle_data,
            install_dir=gemini_install_dir,
            binary_name="gemini",
            user=user,
        )


def _instrumented_gemini_bundle(version: str, platform: SandboxPlatform) -> bytes:
    cache_dir = package_cache_dir("gemini-cli-instrumented-bundles")
    cache_path = cache_dir / (
        "gemini-cli-instrumented-"
        f"{GEMINI_CLI_TRACE_CONTEXT_CACHE_REVISION}-{version}-{platform}.tar.gz"
    )
    if cache_path.exists():
        bundle_data = cache_path.read_bytes()
        _verify_instrumented_gemini_bundle(bundle_data)
        return bundle_data

    stock_bundle = create_npm_bundle(
        package=GEMINI_CLI_TRACE_CONTEXT_CONTRACT.package,
        version=version,
        platform=platform,
        cache_name="gemini-cli-bundles",
    )
    with tempfile.TemporaryDirectory(prefix="inspect-swe-gemini-cli-") as temporary_dir:
        tree = Path(temporary_dir)
        with tarfile.open(fileobj=BytesIO(stock_bundle), mode="r:gz") as archive:
            _extract_npm_bundle(archive, tree)
        patch_gemini_cli_tree(tree, version)
        bundle_data = _archive_npm_tree(tree)

    _verify_instrumented_gemini_bundle(bundle_data)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(bundle_data)
    return bundle_data


def _extract_npm_bundle(archive: tarfile.TarFile, tree: Path) -> None:
    root = tree.resolve()
    for member in archive.getmembers():
        member_path = PurePosixPath(member.name)
        destination = tree / member_path
        if (
            member_path.is_absolute()
            or ".." in member_path.parts
            or not destination.resolve().is_relative_to(root)
        ):
            raise GeminiCliInstrumentationError(
                f"Gemini CLI bundle member escapes staging tree: {member.name}"
            )
        if member.isdir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if member.issym():
            link_destination = destination.parent / member.linkname
            if not link_destination.resolve().is_relative_to(root):
                raise GeminiCliInstrumentationError(
                    f"Gemini CLI bundle link escapes staging tree: {member.name}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(member.linkname)
            continue
        if not (member.isfile() or member.islnk()):
            raise GeminiCliInstrumentationError(
                f"Gemini CLI bundle has unsupported member type: {member.name}"
            )
        source = archive.extractfile(member)
        if source is None:
            raise GeminiCliInstrumentationError(
                f"Gemini CLI bundle cannot read member: {member.name}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)
        destination.chmod(member.mode)


def _archive_npm_tree(tree: Path) -> bytes:
    bundle = BytesIO()
    with tarfile.open(fileobj=bundle, mode="w:gz") as archive:
        for child in tree.iterdir():
            archive.add(child, arcname=child.name)
    return bundle.getvalue()


def _verify_instrumented_gemini_bundle(bundle_data: bytes) -> None:
    target_path = str(GEMINI_CLI_TRACE_CONTEXT_CONTRACT.target_relative_path)
    try:
        with tarfile.open(fileobj=BytesIO(bundle_data), mode="r:gz") as archive:
            target = archive.extractfile(target_path)
            if target is None:
                raise GeminiCliInstrumentationError(
                    "instrumented Gemini CLI cache artifact has no trace-context target"
                )
            target_bytes = target.read()
    except (KeyError, tarfile.TarError) as error:
        raise GeminiCliInstrumentationError(
            "invalid instrumented Gemini CLI cache artifact"
        ) from error

    target_digest = hashlib.sha256(target_bytes).hexdigest()
    if target_digest != GEMINI_CLI_TRACE_CONTEXT_CONTRACT.postimage_sha256:
        raise GeminiCliInstrumentationError(
            "instrumented Gemini CLI cache artifact failed postimage verification"
        )
