import os
from urllib.parse import urlsplit

import anyio
import httpx

# These helpers fetch version pointers, manifests, and multi-megabyte agent
# binaries from external CDNs (github.com, code.kimi.com, storage.googleapis.com).
# httpx's default 5-second read timeout is too tight for some of these endpoints
# from CI runners (code.kimi.com's latest.json alone has breached it), and with
# no retry a single transient blip fails the whole eval — so use a generous
# timeout and retry transient failures with backoff.
_TIMEOUT = httpx.Timeout(60.0, connect=30.0)

# delay before each retry (attempts = len + 1)
_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)

# Release-asset lookups go through the GitHub REST API, which allows 60 requests
# per hour per source IP unauthenticated. Shared CI runners exhaust that budget
# between them, and the resulting `403 rate limit exceeded` is a permanent 4xx as
# far as the retry below is concerned -- so the eval fails outright while
# resolving an agent binary that has nothing to do with the task. A token raises
# the same budget to 5000/hour and is already present in most CI environments.
#
# Host-side only: this authenticates the runner's own metadata request. The token
# is never written into a sandbox, and only the resolved binary crosses that line.
_GITHUB_API_HOSTS = frozenset({"api.github.com"})
_GITHUB_TOKEN_VARS = ("GITHUB_TOKEN", "GH_TOKEN")


def _request_headers(url: str) -> dict[str, str]:
    if urlsplit(url).hostname not in _GITHUB_API_HOSTS:
        return {}
    for var in _GITHUB_TOKEN_VARS:
        token = os.environ.get(var)
        if token:
            return {"Authorization": f"Bearer {token}"}
    return {}


async def download_file(url: str) -> bytes:
    headers = _request_headers(url)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        attempts = len(_RETRY_DELAYS) + 1
        for attempt in range(attempts):
            try:
                response = await client.get(url, follow_redirects=True, headers=headers)
                response.raise_for_status()
                return response.content
            except (httpx.TransportError, httpx.HTTPStatusError) as ex:
                # transport errors (timeouts, resets, DNS) and 5xx responses
                # are transient; 4xx responses are permanent
                permanent = (
                    isinstance(ex, httpx.HTTPStatusError)
                    and ex.response.status_code < 500
                )
                if permanent or attempt == attempts - 1:
                    raise
                await anyio.sleep(_RETRY_DELAYS[attempt])
    raise RuntimeError("unreachable")  # satisfies type checker


async def download_text_file(url: str) -> str:
    return (await download_file(url)).decode("utf-8")
