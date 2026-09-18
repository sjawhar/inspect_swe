"""Splitting for the JSON Lines files and streams the native CLIs produce."""

from collections.abc import Iterator

__all__ = ["jsonl_lines"]


def jsonl_lines(payload: str) -> Iterator[str]:
    r"""Yield each non-empty JSON Lines record in `payload`.

    JSON Lines is delimited by `\n` alone, and a serialiser such as JavaScript's
    `JSON.stringify` leaves U+2028, U+2029 and U+0085 raw inside string values.
    `str.splitlines()` treats all three as line boundaries -- along with VT, FF,
    FS, GS and RS -- so it cuts a record apart mid string and the fragments fail
    to parse. Split on the one delimiter the format actually uses, tolerating
    the blank lines a CRLF producer leaves behind.
    """
    for line in payload.split("\n"):
        record = line[:-1] if line.endswith("\r") else line
        if record:
            yield record
