"""Compatibility with upstream inspect-ai installs.

This fork of inspect_swe normally pairs with the trajectory-labs inspect_ai
fork, but nothing pins that pairing for plain installs: this package's own
dependency floor admits upstream inspect-ai from PyPI, and downstream
projects deliberately leave inspect-ai unpinned. Importing inspect_swe must
therefore work against an upstream inspect-ai; fork-only integration
degrades instead of breaking the import.

``BRIDGE_REQUEST_HEADERS`` is the fork's model-metadata key for bridge
request headers (fork inspect_ai ``model/_model.py``). Upstream inspect-ai
never defines the symbol and its bridge never sets the metadata key, so the
literal fallback keeps ``metadata.get(BRIDGE_REQUEST_HEADERS)`` returning
``None`` there -- headers-dependent behavior switches off, nothing else
changes. The value must match the fork's definition; a unit test asserts
that whenever the fork symbol is present.
"""

import inspect_ai.model as _inspect_ai_model

BRIDGE_REQUEST_HEADERS: str = getattr(
    _inspect_ai_model, "BRIDGE_REQUEST_HEADERS", "bridge_request_headers"
)
