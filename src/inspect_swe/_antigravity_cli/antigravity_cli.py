import json
import re
import shlex
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any, Iterable, Literal, Mapping, Sequence

from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentState,
    BridgedToolsSpec,
    agent,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    GenerateFilter,
    Model,
    ModelResolver,
)
from inspect_ai.scorer import score
from inspect_ai.tool import MCPServerConfig, Skill, install_skills, read_skills
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from .._util._async import is_callable_coroutine
from .._util.agentbinary import ensure_agent_binary_installed
from .._util.centaur import CentaurOptions, CentaurSession, CommandsFilter, run_centaur
from .._util.mcp_ready import DEFAULT_MCP_READY_TIMEOUT, wait_for_mcp_endpoints
from .._util.messages import build_user_prompt
from .._util.path import join_path
from .._util.sandbox import resolve_agent_cwd
from .._util.trace import trace
from .agentbinary import antigravity_cli_binary_source

# Reasoning efforts the CLI accepts. Narrower than most agents' scales: `agy
# --effort` takes exactly these three, and REQUIRES one for the Gemini 3.6/3.7
# Flash families (it exits 1 with "requires --effort" otherwise), because it
# resolves `<model>` + `<effort>` into one catalog id (`gemini-3.6-flash-low`).
AntigravityEffort = Literal["low", "medium", "high"]

# Where the CLI keeps its persistent settings and its global MCP registry. These
# are two different files under two different directories -- settings live with
# the CLI's own state, MCP servers in the shared `~/.gemini/config` tree.
_SETTINGS_DIR = ".gemini/antigravity-cli"
_MCP_CONFIG_DIR = ".gemini/config"


# The CLI states the identity of its own conversation in every primary request:
# official 1.1.27/1.1.28 requests carry exactly one system
# `<user_information>` block containing `Conversation ID: <UUID>`, while the
# title-generation conversation the CLI opens alongside carries no such block.
_USER_INFORMATION_BLOCK = re.compile(
    r"<user_information>(.*?)</user_information>", re.DOTALL
)
_CONVERSATION_ID_LINE = re.compile(r"^[ \t]*Conversation ID:[ \t]*(\S+)[ \t]*$", re.M)
_USER_INFORMATION_DELIMITER = re.compile(r"</?user_information>")


def _framed_blocks(text: str) -> list[str]:
    """Return complete `<user_information>` block bodies from one message."""
    delimiters = _USER_INFORMATION_DELIMITER.findall(text)
    alternating = [
        "<user_information>" if index % 2 == 0 else "</user_information>"
        for index in range(len(delimiters))
    ]
    if len(delimiters) % 2 or delimiters != alternating:
        raise ValueError(
            f"antigravity cli framed its <user_information> with {len(delimiters)} "
            "delimiters that do not open and close complete blocks"
        )
    return _USER_INFORMATION_BLOCK.findall(text)


def _validated_conversation_id(value: str) -> str:
    """Return a UUID conversation id without normalising its printable form."""
    try:
        uuid.UUID(value)
    except ValueError as ex:
        raise ValueError(
            f"antigravity cli declared a malformed conversation id {value!r}"
        ) from ex
    return value


def _native_conversation_id(messages: Sequence[ChatMessage]) -> str | None:
    """Read the CLI conversation id declared in one model request.

    An auxiliary title request carries no system `<user_information>` block.
    Every other framing fault is invalid rather than a signal to guess.
    """
    blocks = [
        block
        for message in messages
        if isinstance(message, ChatMessageSystem)
        for block in _framed_blocks(message.text)
    ]
    if not blocks:
        return None
    if len(blocks) > 1:
        raise ValueError(
            f"antigravity cli declared {len(blocks)} <user_information> blocks "
            "in one request; expected exactly one carrying its conversation id"
        )

    declared = _CONVERSATION_ID_LINE.findall(blocks[0])
    if len(declared) != 1:
        raise ValueError(
            "antigravity cli declared a <user_information> block carrying "
            f"{len(declared)} conversation ids; expected exactly one"
        )
    return _validated_conversation_id(declared[0])


class _NativeConversation:
    """Track the canonical Antigravity conversation for one invocation."""

    def __init__(self, *, unattended: bool, bound_id: str | None = None) -> None:
        self._unattended = unattended
        self.bound_id: str | None = bound_id

    def accept(self, messages: Sequence[ChatMessage]) -> bool:
        """Accept a primary request, rejecting only auxiliary or foreign traffic."""
        conversation_id = _native_conversation_id(messages)
        if conversation_id is None:
            return False
        if self.bound_id is None:
            self.bound_id = conversation_id
        return not self._unattended or self.bound_id == conversation_id


@agent
def antigravity_cli(
    name: str = "Antigravity CLI",
    description: str = dedent("""
       Autonomous coding agent capable of writing, testing, debugging,
       and iterating on code across multiple languages.
    """),
    system_prompt: str | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    mcp_ready_timeout: float = DEFAULT_MCP_READY_TIMEOUT,
    centaur: bool | CentaurOptions = False,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    agy_model: str = "gemini-3.6-flash",
    effort: AntigravityEffort | None = "low",
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    debug: bool | None = None,
    *,
    commands_filter: CommandsFilter | None = None,
    model_resolver: ModelResolver | None = None,
    accumulate_conversations: bool = False,
) -> Agent:
    """Antigravity CLI agent.

    Agent that uses Google's [Antigravity CLI](https://antigravity.google/docs/cli/overview)
    (`agy`) running in a sandbox with Inspect model bridging.

    Model calls are bridged the same way `gemini_cli`'s are: the CLI's direct
    Gemini API route (`modelProvider: "gemini"`, added in `agy` 1.1.13) is
    selected and `GOOGLE_GEMINI_BASE_URL` is pointed at the loopback
    `sandbox_agent_bridge`, so generation never leaves the sandbox and no Google
    sign-in happens. `GEMINI_API_KEY` is set to a placeholder purely to satisfy
    the CLI's credential check -- the bridge does not read it.

    This is a different agent from `antigravity`, which runs the
    `google-antigravity` **SDK** rather than the shipped CLI.

    Use the `attempts` option to enable additional submissions if the initial
    submission(s) are incorrect (by default, no additional attempts are permitted).

    Args:
        name: Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)
        description: Agent description
        system_prompt: Additional system prompt to append
        skills: Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent.
        mcp_servers: MCP servers to make available to the agent
        bridged_tools: Host-side Inspect tools to expose to the agent via MCP
        mcp_ready_timeout: Seconds to wait for bridged MCP endpoints to serve
            tools before the agent launch errors.
        centaur: Run in 'centaur' mode, which makes the Antigravity CLI available to an Inspect `human_cli()` agent rather than running it unattended.
        commands_filter: In centaur mode only, filter or augment the human agent's
            command list (e.g. to add task-specific commands). Ignored outside centaur mode.
        attempts: Configure agent to make multiple attempts
        model: Model name to use for inspect bridge (defaults to main model for task)
        model_aliases: Optional mapping of model names to Model instances or model name strings.
            Allows using custom Model implementations (e.g. wrapped Agents) instead of standard models.
        agy_model: Model name to pass to the CLI. The actual model calls still go
            through the Inspect bridge; this selects the CLI's own client-side
            model configuration (context window, effort handling, tool schema).
        effort: Reasoning effort to pass to the CLI. Passed explicitly by default:
            the Gemini 3.6/3.7 Flash families that `agy` defaults to require it,
            and leaving it implicit would let the CLI's own default decide what
            an eval measured. Pass `None` for a model that rejects the flag.
        filter: Filter for intercepting bridged model requests
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        cwd: Working directory to run the CLI within
        env: Environment variables to set for the CLI
        user: User to execute the CLI with
        sandbox: Optional sandbox environment name
        version: Version of the Antigravity CLI to use. One of:
            - "auto": Use any available version in sandbox, otherwise download latest
            - "sandbox": Use sandbox version (raises RuntimeError if not available)
            - "stable"/"latest": Download and use the latest version
            - "x.x.x": Download and use a specific version
        debug: Trace all debug output.
        model_resolver: Dynamic bridge routing policy called after `model_aliases`
            and before the fallback `model`. Return a model/spec to route, or
            `None` to defer.
        accumulate_conversations: Keep every bridge conversation in
            `state.messages` rather than only the main agent loop.
    """
    # resolve centaur
    if centaur is True:
        centaur = CentaurOptions()

    # resolve model
    model = f"inspect/{model}" if model is not None else "inspect"

    # resolve skills
    resolved_skills = read_skills(skills) if skills is not None else None

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    async def execute(state: AgentState) -> AgentState:
        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "antigravity_cli_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        prior_conversation_id = (
            _native_conversation_id(state.messages) if centaur is False else None
        )
        conversation = _NativeConversation(
            unattended=centaur is False, bound_id=prior_conversation_id
        )

        async with sandbox_agent_bridge(
            state,
            model=model,
            model_aliases=model_aliases,
            filter=filter,
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=bridged_tools,
            model_resolver=model_resolver,
            accumulate_conversations=accumulate_conversations,
            state_filter=conversation.accept,
        ) as bridge:
            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # resolve working directory (home dir if sandbox default is '/')
            agent_cwd = await resolve_agent_cwd(sbox, user, cwd)

            # install the CLI in the sandbox
            agy_binary = await ensure_agent_binary_installed(
                antigravity_cli_binary_source(), version, user, sbox
            )

            # detect sandbox home directory (the CLI resolves both its settings
            # and its global MCP registry relative to $HOME)
            home_result = await sbox.exec(["sh", "-c", "echo $HOME"], user=user)
            sandbox_home = home_result.stdout.strip() or "/root"

            # install skills
            if resolved_skills is not None:
                skills_dir = join_path(agent_cwd, ".agents/skills")
                await install_skills(resolved_skills, sbox, user, skills_dir)

            # mcp servers
            all_mcp_servers = list(mcp_servers or []) + list(bridge.mcp_server_configs)

            settings_dir = join_path(sandbox_home, _SETTINGS_DIR)
            mcp_config_dir = join_path(sandbox_home, _MCP_CONFIG_DIR)
            await sbox.exec(["mkdir", "-p", settings_dir, mcp_config_dir], user=user)
            await sbox.write_file(
                join_path(settings_dir, "settings.json"),
                build_antigravity_settings(unattended=centaur is False),
            )
            await sbox.write_file(
                join_path(mcp_config_dir, "mcp_config.json"),
                build_antigravity_mcp_config(
                    all_mcp_servers, eager_tools=bridge.bridged_tools
                ),
            )

            # build system prompt
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)

            prompt, has_assistant_response = build_user_prompt(state.messages)

            if centaur is False and has_assistant_response:
                if prior_conversation_id is None:
                    raise RuntimeError(
                        "antigravity cli cannot resume: the conversation handed "
                        "to this agent carries assistant turns but no native "
                        "conversation id, so there is no conversation to continue"
                    )

            # Prepend the system prompt to the user prompt: the CLI has no
            # separate --system-prompt flag (same as gemini_cli).
            if system_messages:
                combined_system = "\n\n".join(system_messages)
                prompt = f"{combined_system}\n\n{prompt}"

            cmd = [
                agy_binary,
                "--model",
                agy_model,
                # Omitted only when explicitly disabled: models outside the
                # 3.6/3.7 Flash families reject --effort as not adjustable.
                *(["--effort", effort] if effort is not None else []),
            ]

            # These are print-mode concerns. The human at a Centaur terminal
            # approves actions and reads ordinary output, so it receives none.
            if centaur is False:
                cmd.extend(
                    [
                        "--disable-slash-commands",
                        "--dangerously-skip-permissions",
                        "--output-format",
                        "json",
                    ]
                )
            agent_env = {
                # The CLI's direct-Gemini-API route, pointed at the bridge. Both
                # halves are required: the base URL alone leaves the CLI on its
                # sign-in path, and the key alone leaves generation on Google's
                # endpoint.
                "GOOGLE_GEMINI_BASE_URL": f"http://localhost:{bridge.port}",
                "GEMINI_API_KEY": "api-key",
                # The CLI self-updates from its auto-updater service on startup.
                # Left on, a run silently executes whatever version was stable
                # that day rather than the pinned one -- the same
                # eval-reproducibility hazard as an unpinned download.
                "AGY_CLI_DISABLE_AUTO_UPDATE": "1",
                # No D-Bus in a sandbox, so the CLI's keyring probe has nothing
                # to talk to; the logo art is noise in a captured transcript.
                "AGY_CLI_HIDE_LOGO": "1",
                "HOME": sandbox_home,
            } | (env or {})

            # Gate the launch on the bridged MCP endpoints actually serving
            # tools: the CLI blocks its first turn on MCP connect for headless
            # runs, but only after the endpoint answers `tools/list`.
            _http_mcp_configs = [
                c
                for c in bridge.mcp_server_configs
                if isinstance(c, MCPServerConfigHTTP)
            ]
            if _http_mcp_configs:
                await wait_for_mcp_endpoints(
                    _http_mcp_configs,
                    bridge,
                    sandbox=sandbox,
                    timeout=mcp_ready_timeout,
                    required=True,
                )

            if centaur:
                return await _run_antigravity_cli_centaur(
                    options=centaur,
                    agy_cmd=cmd,
                    agent_env=agent_env,
                    session=CentaurSession(
                        state=bridge.state,
                        invocation=tuple(cmd),
                        environment=agent_env,
                        cwd=agent_cwd,
                        user=user,
                        sandbox=sbox,
                        sandbox_name=sandbox,
                        bridge_port=bridge.port,
                        session_id=None,
                    ),
                    commands_filter=commands_filter,
                )
            else:
                debug_output: list[str] = []
                agent_prompt = prompt
                attempt_count = 0

                while True:
                    agent_cmd = cmd.copy()

                    # Resume by explicit id, never with the CLI's ambient
                    # `--continue` selection.
                    if has_assistant_response or attempt_count > 0:
                        if conversation.bound_id is None:
                            raise RuntimeError(
                                "antigravity cli never declared a conversation "
                                "id, so there is nothing to resume"
                            )
                        agent_cmd.extend(["--conversation", conversation.bound_id])

                    agent_cmd.extend(["--print", agent_prompt])

                    if _http_mcp_configs and attempt_count > 0:
                        await wait_for_mcp_endpoints(
                            _http_mcp_configs,
                            bridge,
                            sandbox=sandbox,
                            timeout=mcp_ready_timeout,
                            required=True,
                        )
                    result = await sbox.exec_remote(
                        cmd=["bash", "-c", 'exec 0</dev/null; "$@"', "bash"]
                        + agent_cmd,
                        options=ExecRemoteAwaitableOptions(
                            cwd=agent_cwd,
                            env=agent_env,
                            user=user,
                            concurrency=False,
                        ),
                        stream=False,
                    )

                    if debug:
                        debug_output.append(result.stdout)
                        debug_output.append(result.stderr)

                    if not result.success:
                        raise RuntimeError(
                            f"Error executing antigravity cli agent {result.returncode}: "
                            f"{_clean_antigravity_error(result.stdout, result.stderr)}"
                        )

                    _verify_native_result(result.stdout, conversation.bound_id)

                    attempt_count += 1
                    if attempt_count >= attempts.attempts:
                        break

                    answer_scores = await score(bridge.state)
                    if attempts.score_value(answer_scores[0].value) == 1.0:
                        break

                    if callable(attempts.incorrect_message):
                        if not is_callable_coroutine(attempts.incorrect_message):
                            raise ValueError(
                                "The incorrect_message function must be async."
                            )
                        agent_prompt = await attempts.incorrect_message(
                            bridge.state, answer_scores
                        )
                    else:
                        agent_prompt = attempts.incorrect_message

                if debug:
                    debug_output.insert(0, "Antigravity CLI Debug Output:")
                    trace("\n".join(debug_output))

        return bridge.state

    return agent_with(execute, name=name, description=description)


def build_antigravity_settings(*, unattended: bool = True) -> str:
    """Build Antigravity CLI settings.json content.

    The direct Gemini route prevents sign-in. Approval policies are present
    only for headless work; Centaur leaves those decisions to the human.
    """
    settings: dict[str, Any] = {
        "modelProvider": "gemini",
        # The CLI's own terminal sandbox requires unprivileged user namespaces,
        # which the enclosing Inspect sandbox commonly denies.
        "enableTerminalSandbox": False,
        "enableTelemetry": False,
        "showTips": False,
        "showFeedbackSurvey": False,
        "altScreenMode": "never",
    }
    if unattended:
        settings["toolPermission"] = "always-proceed"
        settings["artifactReviewPolicy"] = "always-proceed"
    return json.dumps(settings, indent=2)


def build_antigravity_mcp_config(
    mcp_servers: Sequence[MCPServerConfig],
    eager_tools: Mapping[str, Iterable[str]],
) -> str:
    """Build Antigravity CLI mcp_config.json content."""
    servers: dict[str, Any] = {}
    for server in mcp_servers:
        config = server.model_dump(exclude={"name", "tools", "type"}, exclude_none=True)
        if isinstance(server, MCPServerConfigHTTP) and "url" in config:
            config["serverUrl"] = config.pop("url")
        if "cwd" in config and not isinstance(config["cwd"], str):
            config["cwd"] = str(config["cwd"])
        eager = sorted(eager_tools.get(server.name, ()))
        if eager:
            config["tools"] = {name: {"eager": True} for name in eager}
        servers[server.name] = config
    return json.dumps({"mcpServers": servers}, indent=2)


# Opaque base64 payloads the CLI prints on stdout -- Gemini thought signatures,
# emitted one per reasoning turn. They carry no diagnostic text, and a long run
# emits enough of them to fill any error budget, so they are replaced by a short
# placeholder before truncation rather than being allowed to evict the real
# message. 200 chars is well above any base64 token that might carry meaning and
# well below the ~2KB signatures.
_OPAQUE_BLOB = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")
_MAX_ERROR_LEN = 20000


def _clean_antigravity_error(stdout: str, stderr: str) -> str:
    """Trim the CLI's failure output down to something readable in a traceback.

    stderr is placed FIRST and stdout second: the CLI reports its actual failure
    reason on stderr (e.g. "Error: timeout waiting for response") while stdout
    carries the reasoning stream. Concatenating stdout first pushed every real
    error past the truncation limit, so a failed run surfaced as a wall of
    base64 with no cause in it.
    """

    def scrub(text: str) -> str:
        kept = [
            line for line in text.split("\n") if not line.strip().startswith("<think")
        ]
        return _OPAQUE_BLOB.sub(
            lambda match: f"<{len(match.group(0))}-char opaque payload>",
            "\n".join(kept),
        ).strip()

    sections = [
        f"{label}:\n{body}"
        for label, body in (("STDERR", scrub(stderr)), ("STDOUT", scrub(stdout)))
        if body
    ]
    cleaned = "\n\n".join(sections).strip()
    if len(cleaned) > _MAX_ERROR_LEN:
        cleaned = cleaned[:_MAX_ERROR_LEN] + "... (truncated)"
    return cleaned if cleaned else "Unknown error (no output)"


def _native_result_json(stdout: str) -> Mapping[str, Any]:
    """Return the single JSON result emitted by one headless CLI invocation."""
    try:
        parsed: Any = json.loads(stdout.strip())
    except json.JSONDecodeError as ex:
        raise RuntimeError(
            "antigravity cli did not print exactly one JSON result on stdout "
            f"({ex.msg}): {_clean_antigravity_error(stdout, '')}"
        ) from ex

    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"antigravity cli printed a JSON {type(parsed).__name__} rather "
            f"than a result object: {_clean_antigravity_error(stdout, '')}"
        )
    return parsed


def _verify_native_result(stdout: str, conversation_id: str | None) -> None:
    """Require a successful result from the exact conversation being scored."""
    result = _native_result_json(stdout)
    status = result.get("status")
    if status != "SUCCESS":
        raise RuntimeError(
            f"antigravity cli reported status {status!r} for conversation "
            f"{result.get('conversation_id')!r}; expected SUCCESS"
        )

    if conversation_id is None:
        raise RuntimeError(
            "antigravity cli reported success for conversation "
            f"{result.get('conversation_id')!r}, but this run bound no "
            "conversation at all"
        )

    returned = result.get("conversation_id")
    if not isinstance(returned, str):
        raise RuntimeError(
            f"antigravity cli reported success with conversation id {returned!r} "
            f"({type(returned).__name__}) rather than a string, while this run "
            f"bound {conversation_id!r}"
        )
    if returned != conversation_id:
        raise RuntimeError(
            f"antigravity cli returned conversation {returned!r} but this run "
            f"bound {conversation_id!r}; refusing to score a conversation it did not run"
        )


async def _run_antigravity_cli_centaur(
    options: CentaurOptions,
    agy_cmd: list[str],
    agent_env: dict[str, str],
    session: CentaurSession,
    commands_filter: CommandsFilter | None = None,
) -> AgentState:
    instructions = (
        "Antigravity CLI:\n\n"
        " - You may also use the Antigravity CLI via the 'agy' command.\n"
        " - Use 'agy --continue' if you need to resume a previous session."
    )

    # Only the vars the alias needs: exporting HOME would break human_cli.
    centaur_env = {k: v for k, v in agent_env.items() if k != "HOME"}
    agent_env_vars = [f'export {k}="{v}"' for k, v in centaur_env.items()]
    alias_cmd = shlex.join(agy_cmd)
    alias_cmd = "alias agy='" + alias_cmd.replace("'", "'\\''") + "'"
    bashrc = "\n".join(
        agent_env_vars + ["", alias_cmd, f"cd -- {shlex.quote(session.cwd)}"]
    )

    return await run_centaur(
        options, instructions, bashrc, session, commands_filter=commands_filter
    )
