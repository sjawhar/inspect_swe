import json
import posixpath
import shlex
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal, NamedTuple, Sequence

from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentState,
    BridgedToolsSpec,
    agent,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.event import ModelEvent
from inspect_ai.model import (
    BRIDGE_REQUEST_HEADERS,
    ChatMessageSystem,
    GenerateFilter,
    Model,
    ModelResolver,
)
from inspect_ai.scorer import score
from inspect_ai.tool import MCPServerConfig, Skill, install_skills, read_skills
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.util import SandboxEnvironment, store
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from inspect_swe._util._async import is_callable_coroutine
from inspect_swe._util.centaur import (
    CentaurOptions,
    CentaurSession,
    CommandsFilter,
    run_centaur,
)
from inspect_swe._util.mcp_ready import (
    DEFAULT_MCP_READY_TIMEOUT,
    wait_for_mcp_endpoints,
)
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.sandbox import resolve_agent_cwd
from inspect_swe._util.trace import trace

from ._events.consumer import OpenCodeConsumer
from ._events.identity import OpenCodeRequestIdentity, request_identity
from ._events.plugin import (
    OPENCODE_COMPACTION_PLUGIN,
    AppendOnlyCompactionLog,
    compaction_plugin_spec,
)
from .agentbinary import ensure_opencode_setup, seed_opencode_config_dependencies


def _event_identity(event: ModelEvent) -> OpenCodeRequestIdentity | None:
    metadata = event.metadata or {}
    headers = metadata.get(BRIDGE_REQUEST_HEADERS)
    return request_identity(headers) if isinstance(headers, dict) else None


class _OpenCodeConfigPaths(NamedTuple):
    """Separate wrapper-owned bridge state from native configuration inputs."""

    wrapper_dir: str
    native_home: str
    native_global_dir: str


def _opencode_config_paths(
    sandbox_home: str, launch_env: dict[str, str]
) -> _OpenCodeConfigPaths:
    effective_home = launch_env.get("HOME", sandbox_home)
    native_home = launch_env.get("OPENCODE_TEST_HOME", effective_home)
    xdg_config_home = launch_env.get("XDG_CONFIG_HOME", f"{effective_home}/.config")
    return _OpenCodeConfigPaths(
        wrapper_dir=f"{sandbox_home}/.inspect_swe/opencode",
        native_home=native_home,
        native_global_dir=f"{xdg_config_home}/opencode",
    )


async def _native_opencode_config_dirs(
    sandbox: SandboxEnvironment,
    agent_cwd: str,
    global_config_dir: str,
    native_home: str,
    config_dir: str | None,
    project_config_disabled: bool,
    user: str | None,
) -> list[str]:
    """Match OpenCode ConfigPaths.directories before Config.load reaches npm."""
    directories = [global_config_dir]
    if not project_config_disabled:
        worktree_result = await sandbox.exec(
            ["git", "-C", agent_cwd, "rev-parse", "--show-toplevel"], user=user
        )
        worktree = (
            worktree_result.stdout.strip()
            if worktree_result.success
            and worktree_result.stdout.strip().startswith("/")
            else "/"
        )
        project_result = await sandbox.exec(
            [
                "bash",
                "-c",
                dedent("""
                    current="$1"
                    stop="$2"
                    while true; do
                      candidate="$current/.opencode"
                      [ -e "$candidate" ] && printf '%s\n' "$candidate"
                      [ "$current" = "$stop" ] && break
                      parent="$(dirname "$current")"
                      [ "$parent" = "$current" ] && break
                      current="$parent"
                    done
                """),
                "opencode-config-paths",
                agent_cwd,
                worktree,
            ],
            user=user,
        )
        if not project_result.success:
            raise RuntimeError(
                "Unable to discover native OpenCode project config directories: "
                f"{project_result.stderr}"
            )
        directories.extend(
            directory
            for directory in project_result.stdout.splitlines()
            if directory.startswith("/")
        )

    home_config_dir = f"{native_home}/.opencode"
    home_result = await sandbox.exec(
        [
            "bash",
            "-c",
            '[ -e "$1" ] && printf "%s\\n" "$1"; exit 0',
            "opencode-config-home",
            home_config_dir,
        ],
        user=user,
    )
    if not home_result.success:
        raise RuntimeError(
            "Unable to discover the native OpenCode home config directory: "
            f"{home_result.stderr}"
        )
    directories.extend(home_result.stdout.splitlines())
    if config_dir:
        if not config_dir.startswith("/"):
            config_dir = posixpath.normpath(posixpath.join(agent_cwd, config_dir))
        directories.append(config_dir)
    return list(dict.fromkeys(directories))


@agent
def opencode(
    name: str = "OpenCode",
    description: str = dedent("""
       Open-source autonomous coding agent for the terminal, capable
       of writing, testing, debugging, and iterating on code across
       multiple languages.
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
    opencode_model: str = "anthropic/claude-sonnet-4-5",
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
    config_dependency_seed: str | None = None,
) -> Agent:
    """OpenCode agent.

    Agent that uses [OpenCode](https://github.com/anomalyco/opencode)
    running in a sandbox with Inspect model bridging.

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
        centaur: Run in 'centaur' mode, which makes OpenCode available to an Inspect `human_cli()` agent rather than running it unattended.
        commands_filter: In centaur mode only, filter or augment the human agent's
            command list (e.g. to add task-specific commands). Ignored outside centaur mode.
        model_resolver: Dynamic bridge routing policy called after `model_aliases`
            and before the fallback `model`. Return a model/spec to route, or
            `None` to defer.
        accumulate_conversations: Keep every bridge conversation in
            `state.messages` rather than only the main agent loop.
        attempts: Configure agent to make multiple attempts
        model: Model name to use for inspect bridge (defaults to main model for task)
        model_aliases: Optional mapping of model names to Model instances or model name strings.
            Allows using custom Model implementations (e.g., wrapped Agents) instead of standard models.
            When a model name in the mapping is referenced, the corresponding Model/string is used.
        opencode_model: OpenCode model identifier to pass to the CLI in the form
            `provider/model` (default: `"anthropic/claude-sonnet-4-5"`). The actual model
            calls still go through the Inspect bridge; this just selects which provider
            client OpenCode uses to format the request.
        filter: Filter for intercepting bridged model requests
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        cwd: Working directory to run opencode within
        env: Environment variables to set for opencode
        user: User to execute opencode with
        sandbox: Optional sandbox environment name
        version: Version of opencode to use. One of:
            - "auto": Use any available version in sandbox, otherwise download latest
            - "sandbox": Use sandbox version (raises RuntimeError if not available)
            - "stable"/"latest": Download and use the latest version
            - "x.x.x": Download and use a specific version
        debug: Trace all debug output.
        config_dependency_seed: Absolute sandbox directory with the exact CLI
            version's OpenCode config dependency package metadata and production
            node_modules. When omitted, provision an equivalent host-cached seed.
    """
    # resolve centaur
    if centaur is True:
        centaur = CentaurOptions()

    # resolve model
    model = (
        f"inspect/{model}"
        if model is not None
        else "inspect"
        if model_resolver is None
        else None
    )

    # resolve skills
    resolved_skills = read_skills(skills) if skills is not None else None

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    # Keep the requested provider's bridge route configured even when a Centaur
    # operator chooses another supported provider through the bare binary alias.
    provider_id, separator, provider_model_id = opencode_model.partition("/")
    if not separator:
        provider_id = "anthropic"
        provider_model_id = opencode_model

    async def execute(state: AgentState) -> AgentState:
        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "opencode_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        consumer = OpenCodeConsumer(_event_identity)

        async with sandbox_agent_bridge(
            state,
            model=model,
            model_aliases=model_aliases,
            filter=filter,
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=bridged_tools,
            # granted unconditionally to preserve today's behaviour; a grant is
            # inert unless the CLI declares a native web tool
            web_search=True,
            model_resolver=model_resolver,
            accumulate_conversations=accumulate_conversations,
            model_event_metadata_headers=(
                "x-opencode-session",
                "x-session-id",
                "x-parent-session-id",
            ),
            model_event_sink=consumer,
        ) as bridge:
            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # resolve working directory (home dir if sandbox default is '/')
            agent_cwd = await resolve_agent_cwd(sbox, user, cwd)

            # install opencode and its runtime dependencies in sandbox
            opencode_binary, dependency_bin_dirs = await ensure_opencode_setup(
                sbox, version, user
            )

            # combine static mcp configs with bridged tools' mcp servers
            all_mcp_servers = list(mcp_servers or []) + list(bridge.mcp_server_configs)

            # detect sandbox home directory
            home_result = await sbox.exec(["sh", "-c", "echo $HOME"], user=user)
            sandbox_home = home_result.stdout.strip() or "/root"

            launch_env = env or {}
            config_paths = _opencode_config_paths(sandbox_home, launch_env)
            opencode_config_dir = config_paths.wrapper_dir

            # write opencode config to redirect provider requests to the bridge
            # and (optionally) configure mcp servers.
            #
            # The bridge's model-proxy server registers OpenAI-compatible
            # routes (/v1/responses, /v1/chat/completions), the Anthropic
            # Messages route (/v1/messages), and Gemini routes
            # (/v1beta/models/*, /models/*). Each provider client appends its
            # API-relative path to its configured baseURL.
            bridge_url = f"http://localhost:{bridge.port}"
            provider_configs: dict[str, Any] = {
                "anthropic": {"options": {"baseURL": f"{bridge_url}/v1"}}
            }
            if provider_id != "google":
                provider_configs[provider_id] = {
                    "options": {"baseURL": f"{bridge_url}/v1"}
                }
            provider_configs["google"] = {
                "npm": "@ai-sdk/google",
                "options": {
                    "apiKey": "sk-none",
                    "baseURL": f"{bridge_url}/v1beta",
                },
            }
            if provider_id == "google":
                if not provider_model_id:
                    raise ValueError(
                        "opencode_model must name a Google model after 'google/'"
                    )
                provider_configs["google"]["models"] = {
                    provider_model_id: {"name": provider_model_id}
                }
            opencode_config: dict[str, Any] = {
                "$schema": "https://opencode.ai/config.json",
                "model": opencode_model,
                "provider": provider_configs,
            }
            if provider_id == "google":
                # OpenCode's hidden title agent otherwise falls back to the
                # provider's advertised small model. Route that automatic
                # auxiliary request through the caller's selected bridge target.
                opencode_config["agent"] = {"title": {"model": opencode_model}}
            if resolved_skills is not None:
                opencode_config["permission"] = {"skill": {"*": "allow"}}
            if all_mcp_servers:
                opencode_config["mcp"] = resolve_mcp_servers(all_mcp_servers)

            opencode_config_path = f"{opencode_config_dir}/opencode.json"
            plugin_path = f"{opencode_config_dir}/inspect_swe_compaction.mjs"
            event_log_path = f"{opencode_config_dir}/inspect_swe_events.jsonl"
            await sbox.exec(["mkdir", "-p", opencode_config_dir], user=user)
            await sbox.write_file(plugin_path, OPENCODE_COMPACTION_PLUGIN)
            await sbox.write_file(event_log_path, "")
            opencode_config["plugin"] = [
                compaction_plugin_spec(plugin_path, event_log_path)
            ]
            if resolved_skills is not None:
                await install_skills(
                    resolved_skills, sbox, user, f"{opencode_config_dir}/skills"
                )
            await sbox.write_file(opencode_config_path, json.dumps(opencode_config))

            native_config_dirs = await _native_opencode_config_dirs(
                sbox,
                agent_cwd,
                config_paths.native_global_dir,
                config_paths.native_home,
                launch_env.get("OPENCODE_CONFIG_DIR"),
                launch_env.get("OPENCODE_DISABLE_PROJECT_CONFIG", "").lower()
                in {"true", "1"},
                user,
            )
            await seed_opencode_config_dependencies(
                sbox,
                opencode_binary,
                f"{dependency_bin_dirs[0]}/node",
                dependency_bin_dirs,
                native_config_dirs,
                config_dependency_seed,
                user,
            )

            event_log = AppendOnlyCompactionLog()

            async def refresh(_command: str) -> None:
                payload = await sbox.read_file(event_log_path, text=True)
                for native_event in event_log.drain(payload):
                    consumer.on_native_event(native_event)

            # build system prompt (opencode run takes a single positional message
            # and has no separate --system-prompt flag, so we prepend)
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)

            prompt, has_assistant_response = build_user_prompt(state.messages)

            if system_messages:
                combined_system = "\n\n".join(system_messages)
                prompt = f"{combined_system}\n\n{prompt}"

            # base command
            cmd = [
                opencode_binary,
                "run",
                "--model",
                opencode_model,
                "--format",
                "json",
            ]

            # add auto-approve flag only for non-centaur mode
            if centaur is False:
                cmd.append("--dangerously-skip-permissions")

            # setup agent env (add dependencies to PATH so opencode can find them)
            path = ":".join(
                [*dependency_bin_dirs, "/usr/local/bin", "/usr/bin", "/bin"]
            )
            agent_env = {
                # belt-and-braces: set per-provider base URL env vars in addition
                # to the config file. Different opencode provider clients honor
                # different env conventions; the config file is authoritative
                # but env vars don't hurt. The bridge mounts API-specific routes
                # under /v1, so anthropic/openai callers that append "/messages"
                # or "/chat/completions" land on the right handler.
                "ANTHROPIC_BASE_URL": f"{bridge_url}/v1",
                "OPENAI_BASE_URL": f"{bridge_url}/v1",
                "ANTHROPIC_API_KEY": "sk-none",
                "OPENAI_API_KEY": "sk-none",
                "GOOGLE_GENERATIVE_AI_API_KEY": "sk-none",
                "OPENCODE_CONFIG": opencode_config_path,
                "PATH": path,
                "HOME": sandbox_home,
            } | (env or {})

            # Compute bridged HTTP configs once at the outer scope so both the
            # centaur and non-centaur paths gate on the same set. OpenCode's
            # headless mode blocks the first turn on MCP connect, but the
            # endpoint has to be answering `tools/list` first -- this pre-launch
            # gate covers the endpoint half in both centaur and non-centaur modes.
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
                try:
                    return await _run_opencode_centaur(
                        options=centaur,
                        opencode_cmd=cmd,
                        agent_env=agent_env,
                        session=CentaurSession(
                            state=bridge.state,
                            invocation=tuple(cmd),
                            environment=agent_env,
                            cwd=agent_cwd,
                            user=user,
                            sandbox=sbox,
                            bridge_port=bridge.port,
                            session_id=None,
                            refresh=refresh,
                        ),
                        commands_filter=commands_filter,
                    )
                finally:
                    consumer.reset()
            else:
                debug_output: list[str] = []
                agent_prompt = prompt
                attempt_count = 0

                while True:
                    agent_cmd = cmd.copy()

                    # continue previous conversation between attempts (or when
                    # the inbound state already carries an assistant turn)
                    if has_assistant_response or attempt_count > 0:
                        agent_cmd.append("--continue")

                    # add prompt as positional argument at the end
                    agent_cmd.append(agent_prompt)

                    # Retry-loop gate: fires ONLY when this loop is actually
                    # retrying (attempt_count > 0), so the cold-start
                    # pre-centaur gate is not paid for twice on the first
                    # iteration.
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
                    await refresh("opencode execution")

                    if debug:
                        debug_output.append(result.stdout)
                        debug_output.append(result.stderr)

                    if not result.success:
                        cli_error_msg = _clean_opencode_error(
                            result.stdout, result.stderr
                        )
                        raise RuntimeError(
                            f"Error executing opencode agent {result.returncode}: {cli_error_msg}"
                        )

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
                    debug_output.insert(0, "OpenCode Debug Output:")
                    trace("\n".join(debug_output))

        consumer.reset()
        return bridge.state

    return agent_with(execute, name=name, description=description)


def resolve_mcp_servers(
    mcp_servers: Sequence[MCPServerConfig],
) -> dict[str, dict[str, Any]]:
    """Build OpenCode `mcp` config block from MCP server configs.

    OpenCode expects entries keyed by server name with either:
      - {"type": "local", "command": [...], "environment": {...}}
      - {"type": "remote", "url": "...", "headers": {...}}
    """
    out: dict[str, dict[str, Any]] = {}
    for server in mcp_servers:
        config = server.model_dump(exclude={"name", "tools", "type"}, exclude_none=True)
        entry: dict[str, Any] = {"enabled": True}
        if isinstance(server, MCPServerConfigHTTP):
            entry["type"] = "remote"
            if "url" in config:
                entry["url"] = config.pop("url")
            if "headers" in config:
                entry["headers"] = config.pop("headers")
        else:
            entry["type"] = "local"
            # opencode expects the command as a single array including args
            command = config.pop("command", None)
            args = config.pop("args", None)
            if command is None:
                raise ValueError(f"Local MCP server {server.name!r} has no command")
            cmd_list = [command] if isinstance(command, str) else list(command)
            if args:
                cmd_list = cmd_list + list(args)
            entry["command"] = cmd_list
            env_block = config.pop("env", None)
            if env_block:
                entry["environment"] = env_block
        out[server.name] = entry
    return out


def _clean_opencode_error(stdout: str, stderr: str) -> str:
    """Trim OpenCode CLI output to a manageable size for error messages."""
    combined = f"{stdout}\n{stderr}".strip()
    max_len = 2000
    if len(combined) > max_len:
        combined = combined[:max_len] + "... (truncated)"
    return combined if combined else "Unknown error (no output)"


async def _run_opencode_centaur(
    options: CentaurOptions,
    opencode_cmd: list[str],
    agent_env: dict[str, str],
    session: CentaurSession,
    commands_filter: CommandsFilter | None = None,
) -> AgentState:
    instructions = (
        "OpenCode:\n\n"
        " - You may also use OpenCode via the 'opencode' command.\n"
        " - Use 'opencode run --continue' if you need to resume a previous opencode session."
    )

    # build .bashrc content - only export vars needed for the opencode alias,
    # not HOME which would break human_cli (PATH is needed for node)
    centaur_env = {k: v for k, v in agent_env.items() if k != "HOME"}
    agent_env_vars = [f'export {k}="{v}"' for k, v in centaur_env.items()]
    alias_cmd = "alias opencode=" + shlex.quote(opencode_cmd[0])
    bashrc = "\n".join(
        agent_env_vars + ["", alias_cmd, f"cd -- {shlex.quote(session.cwd)}"]
    )

    return await run_centaur(
        options, instructions, bashrc, session, commands_filter=commands_filter
    )
