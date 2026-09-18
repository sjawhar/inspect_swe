import json
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
from inspect_swe._claude_code._events import live_consumer
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


def test_claude_subagent_sidecar_replaying_its_spawn_turn_keeps_root_ownership() -> (
    None
):
    """A subagent sidecar opens by replaying the turn that spawned it.

    Row shape captured from a real session (agent-c#19396): Claude writes one
    assistant message as several records sharing `message.id`, one per content
    block, and `<session>/subagents/agent-*.jsonl` opens with `fork-context-ref`
    followed by a replay of the spawning `tool_use` record -- the parent's
    `message.id`, the child's `agentId`, and a `uuid` of its own. The replay is
    context, so the root keeps the response and the child never waits on it.
    """
    consumer = LiveConsumer()
    spawn = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                id="spawn-response",
                content="",
                tool_calls=[
                    ToolCall(
                        id="task-child",
                        function="Agent",
                        arguments={
                            "description": "audit the store",
                            "subagent_type": "fork",
                        },
                    )
                ],
            )
        ),
    )
    consumer.on_pending(spawn)
    consumer.on_complete(spawn)

    # `refresh()` reads `/subagents/agent-*.jsonl` BEFORE the root session, so
    # the sidecar reaches the consumer first. Feeding the parent first hides
    # this bug entirely, which is how the first fix for it looked correct.
    consumer.process_jsonl_line({"type": "fork-context-ref", "agentId": "child-agent"})
    consumer.process_jsonl_line(
        {
            "uuid": "sidecar-spawn-replay",
            "type": "assistant",
            "agentId": "child-agent",
            "isSidechain": True,
            "message": {
                "id": "spawn-response",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "task-child",
                        "name": "Agent",
                        "input": {"subagent_type": "fork"},
                    }
                ],
            },
        }
    )

    # The root session then writes that one message as several records, one
    # per content block; the sidecar replayed only the tool_use one.
    consumer.process_jsonl_line(
        {
            "uuid": "parent-thinking",
            "type": "assistant",
            "isSidechain": False,
            "message": {"id": "spawn-response", "content": [{"type": "thinking"}]},
        }
    )
    consumer.process_jsonl_line(
        {
            "uuid": "parent-tool-use",
            "type": "assistant",
            "isSidechain": False,
            "message": {
                "id": "spawn-response",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "task-child",
                        "name": "Agent",
                        "input": {"subagent_type": "fork"},
                    }
                ],
            },
        }
    )

    child = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(id="child-response", content="audited")
        ),
    )
    consumer.on_pending(child)
    consumer.on_complete(child)
    consumer.process_jsonl_line(
        {
            "uuid": "sidecar-child-response",
            "type": "assistant",
            "agentId": "child-agent",
            "isSidechain": True,
            "message": {"id": "child-response"},
        }
    )
    consumer.process_jsonl_line(
        {
            "uuid": "child-result",
            "type": "user",
            "toolUseResult": {"agentId": "child-agent"},
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "task-child"}]
            },
        }
    )

    # `refresh()` settles held fork records once the whole drain is read.
    consumer._resolve_fork_context_holds()

    assert child.span_id == "agent-task-child"
    # The spawning turn is the root's work, and the child must not wait on a
    # response it never produced: re-owning it leaves this span open forever.
    assert spawn.span_id is None
    assert consumer._response_agents["spawn-response"] is None
    assert _span_events(SpanEndEvent) == ["agent-task-child"]


def test_claude_fork_first_turn_is_kept_when_no_root_record_claims_it() -> None:
    """A fork whose first sidecar record is its OWN turn keeps it.

    The replay and a genuine first turn are the same shape at the moment they
    arrive -- both sidechain, both carrying the fork's agent ID -- so the
    record is held and settled by what the rest of the drain does. Nothing
    else claims this response, so the child produced it.
    """
    consumer = LiveConsumer()
    spawn = _model_event(
        [],
        ModelOutput.from_message(
            ChatMessageAssistant(
                id="spawn-response",
                content="",
                tool_calls=[
                    ToolCall(
                        id="task-child",
                        function="Agent",
                        arguments={"subagent_type": "fork"},
                    )
                ],
            )
        ),
    )
    consumer.on_pending(spawn)
    consumer.on_complete(spawn)

    consumer.process_jsonl_line({"type": "fork-context-ref", "agentId": "child-agent"})
    consumer.process_jsonl_line(
        {
            "uuid": "sidecar-first-turn",
            "type": "assistant",
            "agentId": "child-agent",
            "isSidechain": True,
            "message": {"id": "child-own-response"},
        }
    )
    consumer.process_jsonl_line(
        {
            "uuid": "child-result",
            "type": "user",
            "toolUseResult": {"agentId": "child-agent"},
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "task-child"}]
            },
        }
    )
    consumer._resolve_fork_context_holds()

    # Unskipped: no root record ever claimed it, so it is the child's.
    assert consumer._response_agents["child-own-response"] == "child-agent"


def test_claude_held_fork_record_still_raises_on_a_genuine_double_claim() -> None:
    """Holding a record must not swallow the conflict the guard exists for.

    Two forks claim one response ID and no root record ever arrives, so
    nothing confirms the held record as a replay. Settling it has to put it
    back through ownership, where the second claimant raises -- discarding it
    because *someone* holds the ID would lose exactly the corruption this
    guard was written to catch.
    """
    consumer = LiveConsumer()
    consumer.process_jsonl_line({"type": "fork-context-ref", "agentId": "agent-a"})
    consumer.process_jsonl_line(
        {
            "uuid": "agent-a-first",
            "type": "assistant",
            "agentId": "agent-a",
            "isSidechain": True,
            "message": {"id": "shared-response", "content": [{"type": "text"}]},
        }
    )
    consumer.process_jsonl_line(
        {
            "uuid": "agent-b-claim",
            "type": "assistant",
            "agentId": "agent-b",
            "isSidechain": True,
            "message": {"id": "shared-response", "content": [{"type": "tool_use"}]},
        }
    )

    with pytest.raises(RuntimeError, match="changed agent ownership"):
        consumer._resolve_fork_context_holds()


def test_claude_non_sidechain_record_after_fork_context_is_never_held() -> None:
    """The window is for sidecar records only.

    A root record that happens to follow a `fork-context-ref` carrying the
    same agent ID is not a replay and must register immediately -- dropping
    the `isSidechain` conjunct has to fail here.
    """
    consumer = LiveConsumer()
    consumer.process_jsonl_line({"type": "fork-context-ref", "agentId": "child-agent"})
    consumer.process_jsonl_line(
        {
            "uuid": "root-record",
            "type": "assistant",
            "agentId": "child-agent",
            "isSidechain": False,
            "message": {"id": "root-response"},
        }
    )

    assert consumer._response_agents["root-response"] == "child-agent"
    assert not consumer._held_fork_records


def test_claude_raises_when_two_native_agents_claim_one_response() -> None:
    """A genuine double claim is still a contract violation.

    The sidecar exemption above is narrow on purpose: it applies only to a
    sidechain record replaying a response the ROOT holds. Two agents naming
    themselves owner of one `message.id` is the corruption the guard exists to
    catch, and must not be swallowed with the replays.
    """
    consumer = LiveConsumer()
    consumer.process_jsonl_line(
        {
            "uuid": "first-claim",
            "type": "assistant",
            "agentId": "agent-one",
            "message": {"id": "shared-response", "content": [{"type": "text"}]},
        }
    )
    with pytest.raises(RuntimeError, match="changed agent ownership"):
        consumer.process_jsonl_line(
            {
                "uuid": "second-claim",
                "type": "assistant",
                "agentId": "agent-two",
                "message": {
                    "id": "shared-response",
                    "content": [{"type": "tool_use"}],
                },
            }
        )

    # Nor does the sidechain flag excuse a claim against another agent: the
    # exemption requires the response to be the root's, not another child's.
    sidechain = LiveConsumer()
    sidechain.process_jsonl_line(
        {
            "uuid": "owning-claim",
            "type": "assistant",
            "agentId": "agent-one",
            "message": {"id": "shared-response"},
        }
    )
    with pytest.raises(RuntimeError, match="changed agent ownership"):
        sidechain.process_jsonl_line(
            {
                "uuid": "sidechain-claim",
                "type": "assistant",
                "agentId": "agent-two",
                "isSidechain": True,
                "message": {"id": "shared-response"},
            }
        )


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
                    "1\t/home/cc/.claude/projects/project/session.jsonl\n"
                    "1\t/home/cc/.claude/projects/project/session/subagents/agent-child.jsonl\n"
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


def _drain_sandbox(
    *,
    paths: list[str],
    contents: dict[str, str],
    read_delay: float = 0.0,
    hang_on: frozenset[str] = frozenset(),
    exec_hangs: bool = False,
    exec_raises: Exception | None = None,
) -> SandboxEnvironment:
    """A sandbox whose enumeration and reads can be slow, dead, or the provider's own error."""
    import anyio

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
            if exec_raises is not None:
                raise exec_raises
            if exec_hangs:
                await anyio.sleep_forever()
            listing = "".join(
                f"{len(contents.get(path, '').encode())}\t{path}\n" for path in paths
            )
            return ExecResult(success=True, returncode=0, stdout=listing, stderr="")

        async def write_file(self, file: str, contents: str | bytes) -> None:
            raise AssertionError(f"unexpected write to {file}")

        @overload
        async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

        @overload
        async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

        async def read_file(self, file: str, text: bool = True) -> str | bytes:
            if file in hang_on:
                await anyio.sleep_forever()
            await anyio.sleep(read_delay)
            return contents[file]

        @classmethod
        async def sample_cleanup(
            cls,
            task_name: str,
            config: SandboxEnvironmentConfigType | None,
            environments: dict[str, SandboxEnvironment],
            interrupted: bool,
        ) -> None:
            return None

    return _Sandbox()


@pytest.mark.anyio
async def test_claude_drain_fails_loudly_when_the_sandbox_never_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A teardown drain against a dead pod must end, and say which step died.

    agent-c#19253: hawk's stop interrupted the sample, the recorder's teardown
    drain then ran `find` in a sandbox whose pod was gone, and the exec never
    returned. The K8s provider's own `timeout=` runs inside the pod, so a dead
    pod never fires it; the bound has to be ours.
    """
    import anyio

    monkeypatch.setattr(live_consumer, "NATIVE_DRAIN_STEP_TIMEOUT_SECONDS", 0.05)
    consumer = LiveConsumer()
    consumer.configure_centaur_session(
        _drain_sandbox(paths=[], contents={}, exec_hangs=True), "cc", "session"
    )

    with anyio.fail_after(5):
        with pytest.raises(
            RuntimeError, match="enumerating session files did not answer"
        ):
            await consumer.refresh("teardown")


@pytest.mark.anyio
async def test_claude_drain_bound_is_per_step_so_a_long_healthy_session_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Many slow-but-answering reads must pass: the bound is on progress, not total work.

    The first version of this fix put one budget over the whole drain. Replaying
    a real 1470-row session with realistic per-read latency took longer than
    that budget and failed on a HEALTHY sandbox. Here every read is far under
    the per-step bound while the drain as a whole is far over it.
    """
    monkeypatch.setattr(live_consumer, "NATIVE_DRAIN_STEP_TIMEOUT_SECONDS", 0.05)
    paths = [f"/home/cc/.claude/projects/project/s{i}.jsonl" for i in range(20)]
    contents = {
        path: f'{{"uuid":"u{i}","type":"assistant","message":{{"id":"r{i}"}}}}'
        for i, path in enumerate(paths)
    }
    consumer = LiveConsumer()
    consumer.configure_centaur_session(
        _drain_sandbox(paths=paths, contents=contents, read_delay=0.02), "cc", None
    )

    await consumer.refresh(
        "score"
    )  # 20 reads x 0.02 s = 0.4 s of work under a 0.05 s step bound

    assert len(consumer._response_agents) == 20


@pytest.mark.anyio
async def test_claude_drain_gives_a_large_file_time_to_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A big session file that is slow but progressing is not a dead pod.

    The provider permits 100 MiB per file, and a fixed per-read wall clock that
    suits a 4 KiB sidecar would fail a legitimate transfer of one that size.
    The read's deadline therefore grows with the size the enumeration reported.
    Here the file takes 4x the floor bound to arrive; sized, it is well inside.
    """
    monkeypatch.setattr(live_consumer, "NATIVE_DRAIN_STEP_TIMEOUT_SECONDS", 0.05)
    # 1 byte per 0.01 s: a 40-byte file may take 0.05 + 0.4 s.
    monkeypatch.setattr(live_consumer, "NATIVE_DRAIN_FLOOR_BYTES_PER_SECOND", 100.0)
    path = "/home/cc/.claude/projects/project/big.jsonl"
    big = '{"uuid":"u","type":"assistant","message":{"id":"r0"}}'.ljust(40)
    consumer = LiveConsumer()
    consumer.configure_centaur_session(
        _drain_sandbox(paths=[path], contents={path: big}, read_delay=0.2), "cc", None
    )

    await consumer.refresh(
        "teardown"
    )  # 0.2 s read against a 0.05 s floor: sized deadline is 0.45 s

    assert "r0" in consumer._response_agents


@pytest.mark.anyio
async def test_claude_drain_names_the_read_that_died_mid_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pod that dies after some files were read fails on that file, by name."""
    import anyio

    monkeypatch.setattr(live_consumer, "NATIVE_DRAIN_STEP_TIMEOUT_SECONDS", 0.05)
    paths = [f"/home/cc/.claude/projects/project/s{i}.jsonl" for i in range(3)]
    contents = {p: '{"uuid":"u","type":"summary"}' for p in paths}
    consumer = LiveConsumer()
    consumer.configure_centaur_session(
        _drain_sandbox(paths=paths, contents=contents, hang_on=frozenset({paths[1]})),
        "cc",
        None,
    )

    # A drain that already read other files may be holding fork records whose
    # resolution depends on files it has not read yet. Seed one so the
    # no-settlement assertions below have something to protect.
    held_record = {"uuid": "held-1", "type": "assistant", "agentId": "agent-held"}
    consumer._held_fork_records["agent-held"] = held_record
    held_before = dict(consumer._held_fork_records)
    with anyio.fail_after(5):
        with pytest.raises(
            RuntimeError, match=rf"reading {paths[1]} \(\d+ bytes\) did not answer"
        ):
            await consumer.refresh("submit")

    # A drain that died mid-way must not settle held fork records: their
    # meaning depends on files it never read. They stay exactly as they were,
    # and the drain is not marked complete.
    assert consumer._held_fork_records == held_before
    assert all(
        consumer._held_fork_records[key] is value for key, value in held_before.items()
    )
    assert not consumer._native_transcript_drained
    assert not consumer._draining_native_session


@pytest.mark.anyio
async def test_claude_drain_leaves_the_providers_own_timeout_error_alone() -> None:
    """The K8s exec raises TimeoutError itself when its in-pod `timeout` fires.

    That is the provider's verdict with its own message and cause; the drain
    must not rename it into ours, or a slow-but-alive pod would be reported as
    dead and the real reason lost.
    """
    provider_error = TimeoutError("Command timed out after 30s. ExecResult(...)")
    consumer = LiveConsumer()
    consumer.configure_centaur_session(
        _drain_sandbox(paths=[], contents={}, exec_raises=provider_error), "cc", None
    )

    with pytest.raises(TimeoutError) as raised:
        await consumer.refresh("score")

    assert raised.value is provider_error


@pytest.mark.anyio
async def test_claude_unpinned_drain_reads_every_session_in_the_sandbox() -> None:
    """An interactive operator may start several `claude` sessions; drain them all.

    The drain used to be configured with one pinned session id and read only
    that file, which is why the operator's `claude` had to be pinned -- and why
    a second one collided. Without a pinned id the enumeration must match every
    root transcript and every sidecar, and each record must be processed.
    """
    commands: list[str] = []

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
            commands.append(cmd[-1])
            return ExecResult(
                success=True,
                returncode=0,
                stdout=(
                    "1\t/home/cc/.claude/projects/project/first.jsonl\n"
                    "1\t/home/cc/.claude/projects/project/second.jsonl\n"
                    "1\t/home/cc/.claude/projects/project/second/subagents/agent-kid.jsonl\n"
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
                "/home/cc/.claude/projects/project/first.jsonl": (
                    '{"type":"summary","summary":"first","leafUuid":"a"}'
                ),
                "/home/cc/.claude/projects/project/second.jsonl": (
                    '{"type":"summary","summary":"second","leafUuid":"b"}'
                ),
                "/home/cc/.claude/projects/project/second/subagents/agent-kid.jsonl": (
                    '{"type":"summary","summary":"kid","leafUuid":"c"}'
                ),
            }[file]

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
    consumer.configure_centaur_session(_Sandbox(), "cc", None)
    seen: list[dict[str, object]] = []
    original = consumer.process_jsonl_line

    def record(raw: dict[str, object], **kwargs: object) -> None:
        seen.append(raw)
        original(raw)

    consumer.process_jsonl_line = record  # type: ignore[method-assign]

    await consumer.refresh("score")

    # No id was pinned, so the enumeration names every session, not one.
    assert "'*.jsonl'" in commands[0]
    assert "'*/*/subagents/agent-*.jsonl'" in commands[0]
    # Sidecar first (it claims the child response), then both roots.
    assert [raw["summary"] for raw in seen] == ["kid", "first", "second"]


@pytest.mark.anyio
async def test_claude_drain_reads_records_carrying_unicode_line_separators() -> None:
    """A record's own text may contain U+2028/U+2029/U+0085; those are not delimiters.

    Claude writes each record with `JSON.stringify`, which leaves those code points
    raw inside string values. Only the newline between records separates them; a
    live session died at `score` because the drain split on them too.
    """
    content = "compacted\u2028then\u2029resumed\u0085here"
    payload = json.dumps(
        {
            "type": "system",
            "uuid": "a2a0d9ee-4f38-4d8b-9a83-6e9c1e9f1d21",
            "subtype": "compact_boundary",
            "content": content,
            "compactMetadata": {"trigger": "manual", "preTokens": 123},
        },
        ensure_ascii=False,
    )
    assert "\u2028" in payload

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
                stdout="1\t/home/cc/.claude/projects/project/only.jsonl\n",
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
            return f"{payload}\n"

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
    consumer.configure_centaur_session(_Sandbox(), "cc", None)

    await consumer.refresh("score")

    compactions = [
        event for event in transcript().events if isinstance(event, CompactionEvent)
    ]
    assert len(compactions) == 1
    assert compactions[0].tokens_before == 123
    assert compactions[0].metadata == {"trigger": "manual", "content": content}


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
