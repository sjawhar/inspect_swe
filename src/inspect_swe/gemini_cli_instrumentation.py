"""Integrity-pinned instrumentation and MCP compliance repair for Gemini CLI 0.58.0."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Final

GEMINI_CLI_INSTRUMENTED_VERSION: Final = "0.58.0"
_PATCH_RESOURCE: Final = "gemini-cli-0.58.0-w3c-trace-context.patch"
_HUNK_HEADER: Final = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_length>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_length>\d+))? @@"
)


class GeminiCliInstrumentationError(RuntimeError):
    """The staged Gemini CLI tree does not meet the instrumentation contract."""


@dataclass(frozen=True)
class GeminiCliTraceContextContract:
    """Immutable source and output identities for the supported Gemini package tree."""

    package: str
    version: str
    upstream_commit: str
    target_relative_path: PurePosixPath
    patch_sha256: str
    preimage_sha256: str
    postimage_sha256: str


GEMINI_CLI_TRACE_CONTEXT_CONTRACT: Final = GeminiCliTraceContextContract(
    package="@google/gemini-cli",
    version=GEMINI_CLI_INSTRUMENTED_VERSION,
    upstream_commit="ac9431c9e2290d68af31a77614ff2fddb2391ca3",
    target_relative_path=PurePosixPath(
        "node_modules/@google/gemini-cli/bundle/chunk-MFLFXOVQ.js"
    ),
    patch_sha256="9ae5330521ddc6ef488eb6d42c8c169d2ae7461e000a16c511faa587966a272f",
    preimage_sha256="5934d3b3bd7fc8ea0c853b9a79e3832b51dda7c2522929d1cc3c03239dea2fde",
    postimage_sha256="a4fa563dbcd792a72f7ce2d03338cbc70f90430b5647cc6b5dcc76c1464a5689",
)
GEMINI_CLI_TRACE_CONTEXT_CACHE_REVISION: Final = (
    f"w3c-trace-context-{GEMINI_CLI_TRACE_CONTEXT_CONTRACT.patch_sha256[:12]}"
)


def patch_gemini_cli_tree(tree: Path, version: str) -> None:
    """Patch an npm install root containing the exact supported Gemini CLI package.

    ``tree`` must contain ``node_modules/@google/gemini-cli``. It is not the
    package directory itself and it is not an OCI root filesystem.
    """
    _require_supported_version(version, GEMINI_CLI_TRACE_CONTEXT_CONTRACT)
    _patch_gemini_cli_tree(
        tree,
        version,
        GEMINI_CLI_TRACE_CONTEXT_CONTRACT,
        _load_patch_resource(),
    )


def _require_supported_version(
    version: str, contract: GeminiCliTraceContextContract
) -> None:
    if version != contract.version:
        raise GeminiCliInstrumentationError(
            "Gemini W3C trace-context instrumentation supports only "
            f"{contract.version}, not {version}"
        )


def _patch_gemini_cli_tree(
    tree: Path,
    version: str,
    contract: GeminiCliTraceContextContract,
    patch_data: bytes,
) -> None:
    _require_supported_version(version, contract)

    package_root = tree / "node_modules" / contract.package
    package_path = package_root / "package.json"
    if not package_path.is_file():
        raise GeminiCliInstrumentationError(
            f"Gemini package metadata is missing: {package_path}"
        )
    try:
        package_data = json.loads(package_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GeminiCliInstrumentationError(
            f"Gemini package metadata is invalid JSON: {package_path}"
        ) from error
    if package_data.get("version") != contract.version:
        raise GeminiCliInstrumentationError(
            "Gemini package metadata does not match the instrumentation version: "
            f"{package_data.get('version')!r} != {contract.version!r}"
        )

    if _sha256(patch_data) != contract.patch_sha256:
        raise GeminiCliInstrumentationError(
            "Gemini instrumentation patch digest mismatch"
        )

    target_path = tree / contract.target_relative_path
    if not target_path.is_file():
        raise GeminiCliInstrumentationError(
            f"Gemini instrumentation target is missing: {target_path}"
        )
    source = target_path.read_bytes()
    if _sha256(source) != contract.preimage_sha256:
        raise GeminiCliInstrumentationError(
            f"Gemini instrumentation target has an unexpected preimage: {target_path}"
        )

    patched = _apply_exact_unified_patch(
        source.decode("utf-8"),
        patch_data.decode("utf-8"),
        contract.target_relative_path,
    ).encode("utf-8")
    if _sha256(patched) != contract.postimage_sha256:
        raise GeminiCliInstrumentationError(
            f"Gemini instrumentation patch produced an unexpected postimage: {target_path}"
        )
    target_path.write_bytes(patched)


def _load_patch_resource() -> bytes:
    return (
        resources.files("inspect_swe._gemini_cli")
        .joinpath(_PATCH_RESOURCE)
        .read_bytes()
    )


def _apply_exact_unified_patch(
    source: str, patch: str, target_path: PurePosixPath
) -> str:
    source_lines = source.splitlines(keepends=True)
    patch_lines = patch.splitlines(keepends=True)
    target = str(target_path)
    expected_headers = (f"--- a/{target}\n", f"+++ b/{target}\n")
    if tuple(patch_lines[:2]) != expected_headers:
        raise GeminiCliInstrumentationError(
            "Gemini instrumentation patch targets the wrong file"
        )

    output: list[str] = []
    source_cursor = 0
    patch_cursor = 2
    hunk_count = 0
    while patch_cursor < len(patch_lines):
        header = patch_lines[patch_cursor]
        match = _HUNK_HEADER.match(header)
        if match is None:
            raise GeminiCliInstrumentationError(
                f"Gemini instrumentation patch has an invalid hunk header: {header!r}"
            )
        if int(match.group("old_start")) < 1:
            raise GeminiCliInstrumentationError(
                f"Gemini instrumentation patch has an invalid hunk position: {header!r}"
            )
        old_length = int(match.group("old_length") or "1")
        new_length = int(match.group("new_length") or "1")
        patch_cursor += 1
        hunk_operations: list[tuple[str, str]] = []
        while patch_cursor < len(patch_lines) and not patch_lines[
            patch_cursor
        ].startswith("@@ "):
            patch_line = patch_lines[patch_cursor]
            if not patch_line or patch_line[0] not in " +-":
                raise GeminiCliInstrumentationError(
                    f"Gemini instrumentation patch has an invalid hunk line: {patch_line!r}"
                )
            hunk_operations.append((patch_line[0], patch_line[1:]))
            patch_cursor += 1
        hunk_old_length = sum(operation != "+" for operation, _ in hunk_operations)
        hunk_new_length = sum(operation != "-" for operation, _ in hunk_operations)
        if hunk_old_length != old_length or hunk_new_length != new_length:
            raise GeminiCliInstrumentationError(
                "Gemini instrumentation patch hunk lengths do not match its header"
            )
        expected_source = [
            expected for operation, expected in hunk_operations if operation != "+"
        ]
        match_starts = [
            start
            for start in range(
                source_cursor, len(source_lines) - len(expected_source) + 1
            )
            if source_lines[start : start + len(expected_source)] == expected_source
        ]
        if len(match_starts) != 1:
            raise GeminiCliInstrumentationError(
                "Gemini instrumentation patch does not match its verified preimage "
                "exactly once after the preceding hunk"
            )
        hunk_start = match_starts[0]
        output.extend(source_lines[source_cursor:hunk_start])
        source_cursor = hunk_start
        for operation, expected in hunk_operations:
            if operation == "+":
                output.append(expected)
            else:
                if source_lines[source_cursor] != expected:
                    raise GeminiCliInstrumentationError(
                        "Gemini instrumentation patch changed while it was being applied"
                    )
                if operation == " ":
                    output.append(source_lines[source_cursor])
                source_cursor += 1
        hunk_count += 1

    if hunk_count == 0:
        raise GeminiCliInstrumentationError("Gemini instrumentation patch has no hunks")
    output.extend(source_lines[source_cursor:])
    return "".join(output)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "GEMINI_CLI_INSTRUMENTED_VERSION",
    "GEMINI_CLI_TRACE_CONTEXT_CACHE_REVISION",
    "GEMINI_CLI_TRACE_CONTEXT_CONTRACT",
    "GeminiCliInstrumentationError",
    "GeminiCliTraceContextContract",
    "patch_gemini_cli_tree",
]
