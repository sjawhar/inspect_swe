from typing import Literal

from inspect_ai import Task, task
from inspect_ai.agent import run
from inspect_ai.dataset import Sample
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import SandboxEnvironmentType
from inspect_swe import claude_code, codex_cli, gemini_cli, opencode

# Delegation is the one agent behaviour no other example produces, and the
# transcript it writes is read by code no other example exercises: a subagent
# gets its own native session file, which the recorder drains alongside the
# root's. A consumer bug in that path is invisible to every other example here
# (agent-c#19396, where the first subagent of any session broke `task score`).
_PROMPT = (
    "Create /workspace/parent.txt containing exactly `parent complete`. "
    "Delegate the creation of /workspace/child.txt, containing exactly "
    "`child complete`, to a subagent rather than writing it yourself. "
    "When both files exist, report that both are complete."
)


@solver
def delegation_solver(
    agent_type: Literal["claude_code", "codex_cli", "gemini_cli", "opencode"],
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        system_prompt = "Delegate sub-tasks to subagents when asked to."
        match agent_type:
            case "claude_code":
                agent = claude_code(system_prompt=system_prompt)
            case "codex_cli":
                agent = codex_cli(system_prompt=system_prompt)
            case "gemini_cli":
                agent = gemini_cli(system_prompt=system_prompt)
            case "opencode":
                agent = opencode(system_prompt=system_prompt)

        agent_state = await run(agent, state.messages)
        state.messages = agent_state.messages
        state.output = agent_state.output
        return state

    return solve


@task
def delegation(
    agent: Literal[
        "claude_code", "codex_cli", "gemini_cli", "opencode"
    ] = "claude_code",
    sandbox: SandboxEnvironmentType | None = "docker",
) -> Task:
    return Task(
        dataset=[Sample(input=_PROMPT)],
        solver=delegation_solver(agent),
        sandbox=sandbox,
    )
