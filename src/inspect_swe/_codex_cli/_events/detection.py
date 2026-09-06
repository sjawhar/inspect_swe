"""Pure helpers for interpreting Codex bridge `ModelEvent`s.

Everything here operates purely on the inspect_ai chat messages / tool calls the
bridge `ModelEventSink` already receives — there is no parsing of Codex's
`--json` stdout stream. This is what lets the consumer reconstruct sub-agent
spans (and detect compaction) bridge-only.
"""

import json
from dataclasses import dataclass
from typing import Any

from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall

# Codex built-in multi-agent tool names.
SPAWN_AGENT = "spawn_agent"
CLOSE_AGENT = "close_agent"
WAIT_AGENT = "wait_agent"

# Codex 0.153.1 `sandboxing/src/spawn.rs` wraps a completed child response in
# an `agent_message` whose first native input_text starts with this header.
FINAL_AGENT_MESSAGE_HEADER = "Message Type: FINAL_ANSWER\nTask name: "

# Marker injected as a user message when Codex performs *local* compaction. Our
# custom bridge provider always forces the local path (remote compaction is gated
# to the real "OpenAI"/Azure providers), so this normal `/v1/responses` call is
# the only compaction signal — and it is what Codex's own tests key on.
# Source: codex-rs/core/templates/compact/prompt.md (injected at compact.rs:70-82).
COMPACTION_MARKER = "You are performing a CONTEXT CHECKPOINT COMPACTION."


@dataclass
class SpawnedAgent:
    """A `spawn_agent` tool-call extracted from a parent's model output."""

    call_id: str
    agent_type: str
    message: str
    reasoning_effort: str | None


def find_spawned_agents(tool_calls: list[ToolCall] | None) -> list[SpawnedAgent]:
    """Spawn_agent tool-calls in a parent's output, with their spawn prompts."""
    result: list[SpawnedAgent] = []
    for tc in tool_calls or []:
        if tc.function != SPAWN_AGENT:
            continue
        args = tc.arguments or {}
        message = args.get("message")
        if not isinstance(message, str) or not message:
            continue
        reasoning = args.get("reasoning_effort")
        result.append(
            SpawnedAgent(
                call_id=tc.id,
                agent_type=str(args.get("agent_type") or "agent"),
                message=message,
                reasoning_effort=str(reasoning) if reasoning else None,
            )
        )
    return result


def find_close_targets(tool_calls: list[ToolCall] | None) -> list[str]:
    """Thread ids targeted by `close_agent` tool-calls in a parent's output."""
    targets: list[str] = []
    for tc in tool_calls or []:
        if tc.function != CLOSE_AGENT:
            continue
        target = (tc.arguments or {}).get("target")
        if isinstance(target, str) and target:
            targets.append(target)
    return targets


@dataclass
class SpawnResult:
    """A native routing key returned by a `spawn_agent` tool result.

    Codex multi-agent v1 returns `agent_id`; v2 returns the slash-prefixed
    `task_name`. Each is an exact native key used by subsequent lifecycle
    traffic, never a reconstructed child identity.
    """

    native_key: str
    nickname: str | None


def spawn_result(message: ChatMessageTool) -> SpawnResult | None:
    """Read a native v1 `agent_id` or v2 `task_name` from a spawn result."""
    if message.function != SPAWN_AGENT:
        return None
    data = _loads(message.text)
    if not isinstance(data, dict):
        return None

    for field in ("agent_id", "task_name"):
        native_key = data.get(field)
        if isinstance(native_key, str) and native_key:
            nickname = data.get("nickname")
            return SpawnResult(
                native_key=native_key,
                nickname=nickname if isinstance(nickname, str) and nickname else None,
            )
    return None


def agent_message_recipients(input_messages: list[ChatMessage]) -> set[str]:
    """V2 child routing keys carried by preserved native `agent_message` input.

    The bridge stores a raw Responses `agent_message` under
    `ContentText.internal["agent_message"]`. Its `recipient` is the native
    v2 task path selected by the parent's `spawn_agent` result.
    """
    recipients: set[str] = set()
    for agent_message in _native_agent_messages(input_messages):
        recipient = agent_message.get("recipient")
        if isinstance(recipient, str) and recipient:
            recipients.add(recipient)
    return recipients


def completed_agent_keys(input_messages: list[ChatMessage]) -> set[str]:
    """Native keys reported completed by tools, notifications, or final handoffs."""
    completed: set[str] = set()
    for msg in input_messages:
        if isinstance(msg, ChatMessageTool):
            if msg.function in (WAIT_AGENT, CLOSE_AGENT):
                _collect_status_completed(_loads(msg.text), completed)
        elif isinstance(msg, ChatMessageUser):
            if "<subagent_notification>" in msg.text:
                _collect_notification_completed(msg.text, completed)

    for agent_message in _native_agent_messages(input_messages):
        if not _is_final_agent_message(agent_message):
            continue
        author = agent_message.get("author")
        if isinstance(author, str) and author:
            completed.add(author)
    return completed


def is_compaction_request(input_messages: list[ChatMessage]) -> bool:
    """Whether this request is a (local) compaction summarization call."""
    return any(
        isinstance(msg, ChatMessageUser)
        and msg.text.lstrip().startswith(COMPACTION_MARKER)
        for msg in input_messages
    )


# ---------------------------------------------------------------------------
# internal
# ---------------------------------------------------------------------------


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _native_agent_messages(input_messages: list[ChatMessage]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in input_messages:
        if not isinstance(message, ChatMessageUser) or not isinstance(
            message.content, list
        ):
            continue
        for content in message.content:
            internal = getattr(content, "internal", None)
            agent_message = (
                internal.get("agent_message") if isinstance(internal, dict) else None
            )
            if (
                isinstance(agent_message, dict)
                and agent_message.get("type") == "agent_message"
            ):
                messages.append(agent_message)
    return messages


def _is_final_agent_message(agent_message: dict[str, Any]) -> bool:
    """Whether Codex's native spawn transport says this author finished."""
    if not isinstance(agent_message.get("id"), str) or not agent_message["id"]:
        return False
    if not isinstance(agent_message.get("recipient"), str) or not agent_message[
        "recipient"
    ]:
        return False
    parts = agent_message.get("content")
    if not isinstance(parts, list) or not parts:
        return False
    first_part = parts[0]
    return (
        isinstance(first_part, dict)
        and first_part.get("type") == "input_text"
        and isinstance(first_part.get("text"), str)
        and first_part["text"].startswith(FINAL_AGENT_MESSAGE_HEADER)
    )


def _collect_status_completed(data: Any, out: set[str]) -> None:
    # {"status": {"<native-key>": {"completed": ...}, ...}}
    if not isinstance(data, dict):
        return
    status = data.get("status")
    if isinstance(status, dict):
        for native_key, value in status.items():
            if (
                isinstance(native_key, str)
                and isinstance(value, dict)
                and "completed" in value
            ):
                out.add(native_key)


def _collect_notification_completed(text: str, out: set[str]) -> None:
    # <subagent_notification>{"agent_path": "<native-key>", "status": {"completed": ...}}</...>
    payload = (
        text.replace("<subagent_notification>", "")
        .replace("</subagent_notification>", "")
        .strip()
    )
    data = _loads(payload)
    if not isinstance(data, dict):
        return
    native_key = data.get("agent_path")
    status = data.get("status")
    if (
        isinstance(native_key, str)
        and isinstance(status, dict)
        and "completed" in status
    ):
        out.add(native_key)
