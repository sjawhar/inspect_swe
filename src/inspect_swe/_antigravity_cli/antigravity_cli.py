import json
import shlex
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal, Sequence

from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentState,
    BridgedToolsSpec,
    agent,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.model import ChatMessageSystem, GenerateFilter, Model
from inspect_ai.scorer import score
from inspect_ai.tool import MCPServerConfig, Skill, install_skills, read_skills
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from .._util._async import is_callable_coroutine
from .._util.agentbinary import ensure_agent_binary_installed
from .._util.centaur import CentaurOptions, run_centaur
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

        async with sandbox_agent_bridge(
            state,
            model=model,
            model_aliases=model_aliases,
            filter=filter,
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=bridged_tools,
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
                join_path(settings_dir, "settings.json"), build_antigravity_settings()
            )
            await sbox.write_file(
                join_path(mcp_config_dir, "mcp_config.json"),
                build_antigravity_mcp_config(all_mcp_servers),
            )

            # build system prompt
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)

            prompt, has_assistant_response = build_user_prompt(state.messages)

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
                "--output-format",
                "text",
                # A task prompt is arbitrary user text: without this, a prompt
                # whose first token looks like `/something` is expanded as a
                # slash command / skill instead of being sent to the model.
                "--disable-slash-commands",
            ]

            # Auto-approve tool calls (permission_mode "always-proceed"). In
            # centaur mode the human at the terminal is the approver, so the
            # flag is withheld -- exactly as gemini_cli withholds --yolo.
            if centaur is False:
                cmd.append("--dangerously-skip-permissions")

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
                "PATH": "/usr/local/bin:/usr/bin:/bin",
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
                await _run_antigravity_cli_centaur(
                    options=centaur,
                    agy_cmd=cmd,
                    agent_env=agent_env,
                    state=state,
                )
            else:
                debug_output: list[str] = []
                agent_prompt = prompt
                attempt_count = 0

                while True:
                    agent_cmd = cmd.copy()

                    # resume previous conversation
                    if has_assistant_response or attempt_count > 0:
                        agent_cmd.append("--continue")

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


def build_antigravity_settings() -> str:
    """Build Antigravity CLI settings.json content.

    `modelProvider` is the load-bearing key: it selects the direct Gemini API
    route (`GEMINI_API_KEY` + `GOOGLE_GEMINI_BASE_URL`) instead of the OAuth
    sign-in the CLI otherwise blocks on. Everything else here removes a way for
    a headless run to stall or to reach outside the sandbox.
    """
    settings: dict[str, Any] = {
        "modelProvider": "gemini",
        # Run tools without prompting. Redundant with
        # --dangerously-skip-permissions for the unattended path, but the flag
        # is withheld in centaur mode where the settings file still applies to
        # anything the human launches.
        "toolPermission": "always-proceed",
        # Headless runs honor the persisted artifact-review policy, and the
        # default ("asks-for-review") is a prompt nobody is there to answer.
        "artifactReviewPolicy": "always-proceed",
        # The CLI's own terminal sandbox. Isolation is the Inspect sandbox's job;
        # the CLI's needs unprivileged user namespaces, which container policy
        # typically denies.
        #
        # The four booleans below MUST be JSON booleans, not the "on"/"off"
        # strings the settings documentation uses for them. A string is dropped
        # on load with no error and no rewrite of the file, so `settings.json`
        # keeps saying "off" while `/config` reports the default -- which is how
        # the first cut of this shipped with telemetry still enabled.
        "enableTerminalSandbox": False,
        # No usage statistics or crash reports off-box from an eval run.
        "enableTelemetry": False,
        "showTips": False,
        "showFeedbackSurvey": False,
        # Sequential stdout rather than the alternate screen buffer: the run is
        # captured as text, and alt-screen escape sequences corrupt it.
        "altScreenMode": "never",
    }
    return json.dumps(settings, indent=2)


def build_antigravity_mcp_config(mcp_servers: Sequence[MCPServerConfig]) -> str:
    """Build Antigravity CLI mcp_config.json content.

    The CLI's remote-server schema names the endpoint `serverUrl`; the `url` and
    `httpUrl` spellings other agents accept are silently ignored, which presents
    as a server that is configured but never connects.
    """
    servers: dict[str, Any] = {}
    for server in mcp_servers:
        config = server.model_dump(exclude={"name", "tools", "type"}, exclude_none=True)
        if isinstance(server, MCPServerConfigHTTP) and "url" in config:
            config["serverUrl"] = config.pop("url")
        if "cwd" in config and not isinstance(config["cwd"], str):
            config["cwd"] = str(config["cwd"])
        servers[server.name] = config
    return json.dumps({"mcpServers": servers}, indent=2)


def _clean_antigravity_error(stdout: str, stderr: str) -> str:
    """Trim the CLI's failure output down to something readable in a traceback."""
    combined = f"{stdout}\n{stderr}"
    cleaned_lines = [
        line for line in combined.split("\n") if not line.strip().startswith("<think")
    ]
    cleaned = "\n".join(cleaned_lines).strip()
    max_len = 2000
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "... (truncated)"
    return cleaned if cleaned else "Unknown error (no output)"


async def _run_antigravity_cli_centaur(
    options: CentaurOptions,
    agy_cmd: list[str],
    agent_env: dict[str, str],
    state: AgentState,
) -> None:
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
    bashrc = "\n".join(agent_env_vars + ["", alias_cmd])

    await run_centaur(options, instructions, bashrc, state)
