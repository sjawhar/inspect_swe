from collections.abc import Iterator
from typing import Literal, overload

import pytest
from inspect_ai._util.content import ContentText
from inspect_ai.event import CompactionEvent, SpanBeginEvent, SpanEndEvent
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import Transcript
from inspect_ai.log._transcript import init_transcript, transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.tool import ToolCall
from inspect_ai.util import (
    ExecResult,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    span,
)
from inspect_swe._claude_code._events.live_consumer import LiveConsumer
from inspect_swe._codex_cli._events.consumer import CodexConsumer
from inspect_swe._codex_cli._events.detection import COMPACTION_MARKER
from pydantic import JsonValue


@pytest.fixture(autouse=True)
def _isolated_transcript() -> Iterator[None]:
    init_transcript(Transcript())
    yield
    init_transcript(Transcript())


def _model_event(
    input_messages: list[ChatMessage],
    output: ModelOutput | None = None,
    *,
    model: str = "mock/model",
    metadata: dict[str, object] | None = None,
) -> ModelEvent:
    return ModelEvent(
        model=model,
        input=input_messages,
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=output or ModelOutput.from_content(model, "done"),
        metadata=metadata,
    )


def _span_events(event_type: type[SpanBeginEvent] | type[SpanEndEvent]) -> list[str]:
    return [
        event.id
        for event in transcript().events
        if isinstance(event, (SpanBeginEvent, SpanEndEvent))
        and isinstance(event, event_type)
    ]


def test_claude_binds_overlapping_child_call_from_native_response_and_agent_ids() -> (
    None
):
    consumer = LiveConsumer()
    spawn = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    ToolCall(
                        id="task-1",
                        function="Agent",
                        arguments={"prompt": "Inspect and report."},
                    ),
                    ToolCall(
                        id="task-2",
                        function="Agent",
                        arguments={"prompt": "Inspect and report."},
                    ),
                ],
            )
        ),
    )

    consumer.on_pending(spawn)
    consumer.on_complete(spawn)
    assert _span_events(SpanBeginEvent) == ["agent-task-1", "agent-task-2"]

    first_child = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                id="response-child-1",
                content="first result",
                tool_calls=[
                    ToolCall(
                        id="child-bash-1",
                        function="Bash",
                        arguments={"command": "printf child"},
                    )
                ],
            )
        ),
    )
    # The first child starts with the parent's model, then switches models
    # before it completes. The second sibling starts from the alternate
    # model too, so no model field can identify either relationship.
    first_child_followup = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(id="response-child-1-followup", content="revised")
        ),
        model="mock/alternate",
    )
    second_child = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(id="response-child-2", content="second result")
        ),
        model="mock/alternate",
    )
    for child in (first_child, first_child_followup, second_child):
        consumer.on_pending(child)
        consumer.on_complete(child)

    # Native responses may persist in either order. Their response IDs, not
    # their prompt or model names, select the child spans.
    consumer.process_jsonl_line(
        {
            "uuid": "native-second-response",
            "type": "assistant",
            "agentId": "native-agent-2",
            "message": {"id": "response-child-2"},
        }
    )
    consumer.process_jsonl_line(
        {
            "uuid": "native-first-followup",
            "type": "assistant",
            "agentId": "native-agent-1",
            "message": {"id": "response-child-1-followup"},
        }
    )
    consumer.process_jsonl_line(
        {
            "uuid": "native-first-response",
            "type": "assistant",
            "agentId": "native-agent-1",
            "message": {"id": "response-child-1"},
        }
    )
    consumer.process_jsonl_line(
        {
            "uuid": "native-first-result",
            "type": "user",
            "toolUseResult": {"agentId": "native-agent-1"},
            "message": {"content": [{"type": "tool_result", "tool_use_id": "task-1"}]},
        }
    )
    consumer.process_jsonl_line(
        {
            "uuid": "native-second-result",
            "type": "user",
            "toolUseResult": {"agentId": "native-agent-2"},
            "message": {"content": [{"type": "tool_result", "tool_use_id": "task-2"}]},
        }
    )

    assert first_child.span_id == "agent-task-1"
    assert first_child_followup.span_id == "agent-task-1"
    assert second_child.span_id == "agent-task-2"
    assert (first_child.model, first_child_followup.model, second_child.model) == (
        "mock/model",
        "mock/alternate",
        "mock/alternate",
    )
    assert _span_events(SpanEndEvent) == ["agent-task-1", "agent-task-2"]


@pytest.mark.anyio
async def test_claude_completion_drain_preserves_stdout_child_parent_owner() -> None:
    class _Sandbox(SandboxEnvironment):
        async def exec(
            self,
            cmd: list[str],
            input: str | bytes | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            user: str | None = None,
            timeout: int | None = None,
            timeout_retry: bool = True,
            concurrency: bool = True,
        ) -> ExecResult[str]:
            return ExecResult(
                success=True,
                returncode=0,
                stdout=(
                    "/home/cc/.claude/projects/project/session.jsonl\n"
                    "/home/cc/.claude/projects/project/session/subagents/agent-child.jsonl\n"
                ),
                stderr="",
            )

        async def write_file(self, file: str, contents: str | bytes) -> None:
            raise AssertionError(f"unexpected write to {file}")

        @overload
        async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

        @overload
        async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

        async def read_file(self, file: str, text: bool = True) -> str | bytes:
            if not text:
                raise AssertionError(f"unexpected binary read of {file}")
            return {
                "/home/cc/.claude/projects/project/session.jsonl": """
{"type":"queue-operation","operation":"enqueue","timestamp":"2026-09-06T07:29:26.931Z","sessionId":"39d133c6-e80d-4fb1-b9d4-8fe5c7a4948b","content":"Create `/workspace/parent.txt` containing exactly `parent complete` and delegate creation of `/workspace/child.txt` containing exactly `child complete`. Then report that both files are complete.\\n"}
{"type":"queue-operation","operation":"dequeue","timestamp":"2026-09-06T07:29:26.933Z","sessionId":"39d133c6-e80d-4fb1-b9d4-8fe5c7a4948b"}
{"type":"atis-latch","sessionId":"39d133c6-e80d-4fb1-b9d4-8fe5c7a4948b","atis":""}
{"uuid":"root-tool-result","type":"user","toolUseResult":{"agentId":"child-agent-id"},"message":{"content":[{"type":"tool_result","tool_use_id":"task-child"}]}}
{"uuid":"child-response","type":"assistant","message":{"id":"response-child-bash"}}
""",
                "/home/cc/.claude/projects/project/session/subagents/agent-child.jsonl": """
{"parentUuid":"5a6098d8-e0af-423e-a1e3-d9dfa553247c","isSidechain":true,"agentId":"child-agent-id","attachment":{"type":"skill_listing","content":"- dataviz: Use this skill whenever you create a chart."}}
{"uuid":"child-response","type":"assistant","agentId":"child-agent-id","message":{"id":"response-child-bash"}}
""",
            }[file].strip()

        @classmethod
        async def sample_cleanup(
            cls,
            task_name: str,
            config: SandboxEnvironmentConfigType | None,
            environments: dict[str, SandboxEnvironment],
            interrupted: bool,
        ) -> None:
            return None

    consumer = LiveConsumer()
    consumer.process_jsonl_line(
        {
            "parentUuid": "5a6098d8-e0af-423e-a1e3-d9dfa553247c",
            "isSidechain": True,
            "agentId": "child-agent-id",
            "attachment": {
                "type": "skill_listing",
                "content": "- dataviz: Use this skill whenever you create a chart.",
            },
        }
    )
    consumer.configure_centaur_session(_Sandbox(), "cc", "session")
    spawn = _model_event(
        [],
        ModelOutput.for_tool_call(
            "mock/model",
            "Agent",
            {"description": "write the child proof file"},
            tool_call_id="task-child",
        ),
    )
    consumer.on_pending(spawn)
    consumer.on_complete(spawn)

    # Claude stdout emits the child response before native completion drain.
    # Its exact parent tool-use ID must own that response even though it has no
    # isSidechain flag or agent ID, and therefore consumes the UUID that the
    # sidecar would otherwise use.
    consumer.process_jsonl_line(
        {
            "uuid": "child-response",
            "type": "assistant",
            "parent_tool_use_id": "task-child",
            "message": {"id": "response-child-bash"},
        },
        from_stdout=True,
    )

    child = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                id="response-child-bash",
                content="",
                tool_calls=[
                    ToolCall(
                        id="child-bash",
                        function="Bash",
                        arguments={"command": "printf child"},
                    )
                ],
            )
        ),
    )
    consumer.on_pending(child)
    consumer.on_complete(child)

    # The root stream can reach its native tool_result before completion drain.
    # The later sidecar is intentionally deduplicated by the earlier stdout
    # UUID, so the child association must remain exact without agentId.
    consumer.process_jsonl_line(
        {
            "uuid": "root-tool-result",
            "type": "user",
            "toolUseResult": {"agentId": "child-agent-id"},
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "task-child"}]
            },
        }
    )
    await consumer.drain_completion()

    assert child.span_id == "agent-task-child"

    assert _span_events(SpanEndEvent) == ["agent-task-child"]

    consumer.reset()
    assert _span_events(SpanEndEvent) == ["agent-task-child"]


@pytest.mark.parametrize(
    "raw",
    (
        {"type": "future-session-record"},
        {
            "parentUuid": "5a6098d8-e0af-423e-a1e3-d9dfa553247c",
            "isSidechain": True,
            "agentId": "child-agent-id",
            "attachment": {"type": "future_listing", "content": "new state"},
        },
        {"type": "future-session-record", "uuid": "metadata-event"},
    ),
)
def test_claude_skips_non_conversation_session_records(raw: dict[str, object]) -> None:
    LiveConsumer().process_jsonl_line(raw)

    assert transcript().events == []


@pytest.mark.parametrize(
    "raw",
    (
        {"type": "user"},
        {"type": "assistant"},
        {"type": "system", "subtype": "compact_boundary"},
    ),
)
def test_claude_rejects_uuid_less_conversation_records(
    raw: dict[str, object],
) -> None:
    with pytest.raises(
        RuntimeError, match="Claude session event is missing its native UUID."
    ):
        LiveConsumer().process_jsonl_line(raw)


def test_claude_reset_preserves_unresolved_child_after_cancelled_attempt() -> None:
    consumer = LiveConsumer()
    spawn = _model_event(
        [],
        ModelOutput.for_tool_call(
            "mock/model",
            "Agent",
            {"description": "write the child proof file"},
            tool_call_id="task-child",
        ),
    )
    consumer.on_pending(spawn)
    consumer.on_complete(spawn)

    child = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(id="response-child", content="cancelled")
        ),
    )
    consumer.on_pending(child)
    consumer.on_complete(child)

    consumer.reset()

    assert child.span_id is None
    assert child in transcript().events
    assert _span_events(SpanEndEvent) == ["agent-task-child"]


def test_claude_records_native_child_compaction_after_identity_binding() -> None:
    consumer = LiveConsumer()
    spawn = _model_event(
        [],
        ModelOutput.for_tool_call(
            "mock/model",
            "Agent",
            {"prompt": "Inspect the repository."},
            tool_call_id="task-1",
        ),
    )
    consumer.on_pending(spawn)
    consumer.on_complete(spawn)

    consumer.process_jsonl_line(
        {
            "uuid": "native-child-compaction",
            "type": "system",
            "subtype": "compact_boundary",
            "agentId": "native-agent-1",
            "compactMetadata": {"trigger": "manual", "preTokens": 42},
        }
    )
    consumer.process_jsonl_line(
        {
            "uuid": "native-child-result",
            "type": "user",
            "toolUseResult": {"agentId": "native-agent-1"},
            "message": {"content": [{"type": "tool_result", "tool_use_id": "task-1"}]},
        }
    )

    events = [
        event for event in transcript().events if isinstance(event, CompactionEvent)
    ]
    assert len(events) == 1
    assert events[0].span_id == "agent-task-1"
    assert events[0].metadata == {
        "trigger": "manual",
        "content": "Conversation compacted",
    }


def _codex_spawn(call_id: str, message: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function="spawn_agent",
        arguments={"agent_type": "explorer", "message": message},
    )


def _codex_v2_spawn(call_id: str, task_name: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function="spawn_agent",
        arguments={
            "task_name": task_name.removeprefix("/root/"),
            "message": "Create the child proof file.",
        },
    )


@pytest.mark.anyio
async def test_codex_does_not_merge_overlapping_child_prompts_into_parent() -> None:
    consumer = CodexConsumer()
    parent = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    _codex_spawn("spawn-1", "Inspect the repository and report."),
                    _codex_spawn("spawn-2", "Inspect the repository and report."),
                ],
            )
        ),
    )

    async with span("human_cli", type="agent", id="human-cli"):
        consumer.on_pending(parent)
        consumer.on_complete(parent)
        child = _model_event(
            [ChatMessageUser(content="Inspect the repository and report.")]
        )
        consumer.on_pending(child)
        assert child.span_id is None
        consumer.reset()


@pytest.mark.anyio
async def test_codex_buffers_child_until_late_spawn_result_and_keeps_root_continuation() -> (
    None
):
    consumer = CodexConsumer()
    task_name = "/root/write_child_proof"
    root_metadata: dict[str, object] = {
        "agent_bridge": {"codex": {"thread_id": "thread-root"}}
    }
    parent = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                content="",
                tool_calls=[_codex_v2_spawn("spawn-child", task_name)],
            )
        ),
        metadata=root_metadata,
    )
    child = _model_event(
        [
            ChatMessageUser(
                content=[
                    ContentText(
                        text="Agent message from /root:\nCreate the child proof file.",
                        internal={
                            "agent_message": {
                                "type": "agent_message",
                                "recipient": task_name,
                            }
                        },
                    )
                ]
            )
        ],
        ModelOutput.from_message(
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    ToolCall(
                        id="child-exec",
                        function="exec_command",
                        arguments={"cmd": "printf child"},
                    )
                ],
            )
        ),
    )
    root_continuation = _model_event(
        [
            ChatMessageTool(
                content='{"task_name": "/root/write_child_proof"}',
                tool_call_id="spawn-child",
                function="spawn_agent",
            ),
            ChatMessageUser(content="Continue coordinating the investigation."),
        ],
        ModelOutput.from_message(
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    ToolCall(
                        id="root-exec",
                        function="exec_command",
                        arguments={"cmd": "printf parent"},
                    )
                ],
            )
        ),
        metadata=root_metadata,
    )
    compaction = _model_event(
        [ChatMessageUser(content=COMPACTION_MARKER)],
        metadata=root_metadata,
    )

    async with span("human_cli", type="agent", id="human-cli"):
        consumer.on_pending(parent)
        consumer.on_complete(parent)
        consumer.on_pending(child)
        consumer.on_complete(child)
        assert child not in transcript().events

        # Codex can start and finish the child call before the root receives
        # the exact spawn result. The root continuation keeps the outer span
        # while releasing the child event after its native binding arrives.
        consumer.on_pending(root_continuation)
        consumer.on_complete(root_continuation)
        consumer.on_pending(compaction)
        consumer.on_complete(compaction)

        assert child.span_id == "agent-spawn-child"
        compactions = [
            event for event in transcript().events if isinstance(event, CompactionEvent)
        ]
        assert [event.span_id for event in compactions] == ["human-cli"]
        assert root_continuation.span_id == "human-cli"
        model_events = [
            event for event in transcript().events if isinstance(event, ModelEvent)
        ]
        assert model_events == [parent, child, root_continuation, compaction]
        consumer.reset()


@pytest.mark.anyio
async def test_codex_rejects_multi_recipient_bootstrap_without_exact_single_span() -> (
    None
):
    consumer = CodexConsumer()
    root_metadata: dict[str, object] = {
        "agent_bridge": {"codex": {"thread_id": "thread-root"}}
    }
    first_path = "/root/write_first_proof"
    second_path = "/root/write_second_proof"
    parent = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    _codex_v2_spawn("spawn-first", first_path),
                    _codex_v2_spawn("spawn-second", second_path),
                ],
            )
        ),
        metadata=root_metadata,
    )

    def handoff(task_name: str) -> ChatMessageUser:
        return ChatMessageUser(
            content=[
                ContentText(
                    text=f"Agent message from /root:\n{task_name}",
                    internal={
                        "agent_message": {
                            "type": "agent_message",
                            "recipient": task_name,
                        }
                    },
                )
            ]
        )

    ambiguous_child = _model_event([handoff(first_path), handoff(second_path)])
    root_continuation = _model_event(
        [
            ChatMessageTool(
                content=f'{{"task_name": "{first_path}"}}',
                tool_call_id="spawn-first",
                function="spawn_agent",
            ),
            ChatMessageTool(
                content=f'{{"task_name": "{second_path}"}}',
                tool_call_id="spawn-second",
                function="spawn_agent",
            ),
        ],
        metadata=root_metadata,
    )

    async with span("human_cli", type="agent", id="human-cli"):
        consumer.on_pending(parent)
        consumer.on_complete(parent)
        consumer.on_pending(ambiguous_child)
        consumer.on_complete(ambiguous_child)
        consumer.on_pending(root_continuation)

        assert root_continuation.span_id == "human-cli"
        assert ambiguous_child not in transcript().events
        with pytest.raises(
            RuntimeError,
            match="Codex child events were not bound to an exact native spawn result",
        ):
            consumer.reset()
        assert _span_events(SpanEndEvent) == [
            "agent-spawn-second",
            "agent-spawn-first",
        ]


@pytest.mark.anyio
async def test_codex_binds_same_model_v2_siblings_from_native_agent_message_recipients() -> (
    None
):
    consumer = CodexConsumer()
    first_path = "/root/write_child_proof"
    second_path = "/root/write_second_proof"
    parent = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    _codex_v2_spawn("spawn-1", first_path),
                    _codex_v2_spawn("spawn-2", second_path),
                ],
            )
        ),
    )
    binding = _model_event(
        [
            ChatMessageTool(
                content='{"task_name": "/root/write_second_proof"}',
                tool_call_id="spawn-2",
                function="spawn_agent",
            ),
            ChatMessageTool(
                content='{"task_name": "/root/write_child_proof"}',
                tool_call_id="spawn-1",
                function="spawn_agent",
            ),
        ]
    )

    def child_event(task_path: str, agent_message_id: str) -> ModelEvent:
        handoff: dict[str, JsonValue] = {
            "type": "agent_message",
            "id": agent_message_id,
            "author": "/root",
            "recipient": task_path,
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Message Type: NEW_TASK\n"
                        f"Task name: {task_path}\n"
                        "Sender: /root\n"
                        "Payload:\n"
                    ),
                },
                {
                    "type": "encrypted_content",
                    "encrypted_content": "Create the child proof file.",
                },
            ],
        }
        return _model_event(
            [
                ChatMessageUser(
                    content=[
                        ContentText(
                            text="Agent message from /root:\nCreate the child proof file.",
                            internal={"agent_message": handoff},
                        )
                    ]
                )
            ]
        )

    first_child = child_event(first_path, "amsg_01a07523-ade2-7363-949c-08a1e028863a")
    second_child = child_event(second_path, "amsg_01a07523-ade2-7363-949c-08a1e028863b")
    completion = _model_event(
        [
            ChatMessageUser(
                content=[
                    ContentText(
                        text=(
                            "Agent message from /root/write_child_proof:\n"
                            "The child proof file is complete."
                        ),
                        internal={
                            "agent_message": {
                                "type": "agent_message",
                                "id": "amsg_01a0752c-b962-7db3-8566-816a80a74092",
                                "author": first_path,
                                "recipient": "/root",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": (
                                            "Message Type: FINAL_ANSWER\n"
                                            "Task name: /root\n"
                                            "Sender: /root/write_child_proof\n"
                                            "Payload:\n"
                                            "The child proof file is complete."
                                        ),
                                    }
                                ],
                            }
                        },
                    )
                ]
            )
        ]
    )

    async with span("human_cli", type="agent", id="human-cli"):
        consumer.on_pending(parent)
        consumer.on_complete(parent)
        consumer.on_pending(binding)
        assert binding.span_id is None
        # The native child requests may arrive in either order.
        consumer.on_pending(second_child)
        consumer.on_pending(first_child)
        consumer.on_pending(completion)

    # All three calls use mock/model. Only the exact task_name/recipient join,
    # not model or prompt similarity, selects the sibling spans.
    assert parent.model == first_child.model == second_child.model == "mock/model"
    assert first_child.span_id == "agent-spawn-1"
    assert second_child.span_id == "agent-spawn-2"
    assert [
        event for event in _span_events(SpanBeginEvent) if event.startswith("agent-")
    ] == ["agent-spawn-1", "agent-spawn-2"]
    assert [
        event for event in _span_events(SpanEndEvent) if event.startswith("agent-")
    ] == ["agent-spawn-1"]
    consumer.reset()
    assert [
        event for event in _span_events(SpanEndEvent) if event.startswith("agent-")
    ] == ["agent-spawn-1", "agent-spawn-2"]


def test_codex_reset_closes_unbound_child_once() -> None:
    consumer = CodexConsumer()
    spawn = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                content="", tool_calls=[_codex_spawn("spawn-1", "Inspect this.")]
            )
        ),
    )

    consumer.on_pending(spawn)
    consumer.on_complete(spawn)
    consumer.reset()
    consumer.reset()

    assert _span_events(SpanEndEvent) == ["agent-spawn-1"]
