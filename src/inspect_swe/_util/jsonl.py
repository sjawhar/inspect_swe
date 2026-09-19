"""Splitting for the JSON Lines files and streams the native CLIs produce."""

from collections.abc import Iterator

__all__ = ["jsonl_lines", "jsonl_records"]


def jsonl_records(payload: str) -> Iterator[tuple[str, bool]]:
    r"""Yield each non-empty JSON Lines record in `payload` with whether a newline closed it.

    JSON Lines is delimited by `\n` alone, and a serialiser such as JavaScript's
    `JSON.stringify` leaves U+2028, U+2029 and U+0085 raw inside string values.
    `str.splitlines()` treats all three as line boundaries -- along with VT, FF,
    FS, GS and RS -- so it cuts a record apart mid string and the fragments fail
    to parse. Split on the one delimiter the format actually uses, tolerating
    the blank lines a CRLF producer leaves behind.

    The second element is False only for a final segment that no newline closed.
    On a file another process is still appending to, that segment is a write in
    flight, not a record: a reader that parses it sees a truncated document.
    """
    lines = payload.split("\n")
    last = len(lines) - 1
    for index, line in enumerate(lines):
        record = line[:-1] if line.endswith("\r") else line
        if record:
            yield record, index != last


def jsonl_lines(payload: str) -> Iterator[str]:
    """Yield each non-empty JSON Lines record in `payload`; see `jsonl_records`."""
    for record, _ in jsonl_records(payload):
        yield record
