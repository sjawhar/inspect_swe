"""Tests for generic in-process version-resolution caching."""

from collections.abc import Awaitable, Callable, Iterator

import anyio
import pytest
from inspect_swe._util import versioncache
from inspect_swe._util.versioncache import cached_version_resolution


@pytest.fixture(autouse=True)
def clear_version_cache() -> Iterator[None]:
    versioncache._resolved_versions.clear()
    versioncache._failed_resolutions.clear()
    yield
    versioncache._resolved_versions.clear()
    versioncache._failed_resolutions.clear()


def test_cached_version_resolution_resolves_once() -> None:
    calls = 0

    async def resolve() -> str:
        nonlocal calls
        calls += 1
        return "1.2.3"

    async def run() -> None:
        assert await cached_version_resolution("agent", resolve) == "1.2.3"
        assert await cached_version_resolution("agent", resolve) == "1.2.3"

    anyio.run(run)
    assert calls == 1


def test_cached_version_resolution_keys_are_independent() -> None:
    async def run() -> None:
        assert await cached_version_resolution("a", _const("1.0.0")) == "1.0.0"
        assert await cached_version_resolution("b", _const("2.0.0")) == "2.0.0"
        # each key keeps its own entry
        assert await cached_version_resolution("a", _const("9.9.9")) == "1.0.0"

    anyio.run(run)


def _const(value: str) -> Callable[[], Awaitable[str]]:
    async def resolve() -> str:
        return value

    return resolve
