"""Supported OpenCode plugin used to preserve native compaction events."""

import json

OPENCODE_COMPACTION_PLUGIN = r"""import { appendFileSync } from "node:fs"

export default async function inspectSWECompactionPlugin(_input, options) {
  const eventLog = options?.eventLog
  if (typeof eventLog !== "string" || eventLog.length === 0) {
    throw new Error("inspect_swe OpenCode compaction plugin requires eventLog")
  }

  return {
    event: async ({ event }) => {
      if (event.type !== "session.compacted") return
      const sessionID = event.properties?.sessionID
      if (typeof sessionID !== "string" || sessionID.length === 0) {
        throw new Error("OpenCode session.compacted event is missing sessionID")
      }
      appendFileSync(eventLog, JSON.stringify({ type: event.type, properties: { sessionID } }) + "\\n", "utf8")
    },
  }
}
"""


class AppendOnlyCompactionLog:
    """Drain each complete JSONL plugin record without truncating its producer."""

    def __init__(self) -> None:
        self._offset = 0

    def drain(self, payload: str) -> list[dict[str, object]]:
        """Parse precisely the bytes appended since the prior successful drain."""
        if len(payload) < self._offset:
            raise RuntimeError("OpenCode native event log is no longer append-only.")
        new_payload = payload[self._offset :]
        events = compaction_events(new_payload)
        self._offset = len(payload)
        return events


def compaction_events(payload: str) -> list[dict[str, object]]:
    """Parse the newline-delimited records emitted by the configured plugin."""
    events: list[dict[str, object]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Invalid OpenCode native event at line {line_number}."
            ) from error
        if not isinstance(event, dict):
            raise RuntimeError(
                f"OpenCode native event at line {line_number} is not an object."
            )
        native_event: dict[str, object] = {}
        for key, value in event.items():
            if not isinstance(key, str):
                raise RuntimeError(
                    f"OpenCode native event at line {line_number} has a non-string key."
                )
            native_event[key] = value
        events.append(native_event)
    return events


def compaction_plugin_spec(
    plugin_path: str, event_log_path: str
) -> tuple[str, dict[str, str]]:
    """Return OpenCode's supported plugin spec and its private event-log option."""
    return (plugin_path, {"eventLog": event_log_path})
