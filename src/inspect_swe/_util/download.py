import os
from urllib.parse import urlsplit

import httpx

# Release-asset lookups go through the GitHub REST API, which allows 60 requests per hour
# per source IP unauthenticated. Shared CI runners exhaust that budget between them, and
# the resulting `403 rate limit exceeded` fails the eval while resolving an agent binary
# that has nothing to do with the task under test. A token raises the same budget to
# 5000/hour and is already present in most CI environments.
#
# Host-side only: this authenticates the runner's own metadata request. The token is never
# written into a sandbox -- only the resolved binary crosses that line.
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
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url, follow_redirects=True, headers=_request_headers(url)
        )
        response.raise_for_status()
        return response.content


async def download_text_file(url: str) -> str:
    return (await download_file(url)).decode("utf-8")
