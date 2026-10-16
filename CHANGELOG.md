# Changelog

All notable changes to tinyharness. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/): while on 0.x, minor versions may break
the SDK, and every break is listed here.

## Unreleased

### Added
- ChatGPT Plus and Pro users can choose the separate ChatGPT plan provider and use
  Continue with ChatGPT instead of creating an OpenAI API key. tinyharness validates
  the OpenID identity, stores renewable credentials privately, discovers models for
  the selected account, and follows the plan-usage Responses API restrictions.
- GitHub Copilot subscriptions can connect through GitHub's device sign-in. tinyharness
  loads the account's live Copilot model catalog and uses each model's advertised Chat,
  Responses or Anthropic Messages endpoint while keeping the tinyharness agent loop.
- LiteLLM is a first-class gateway choice. Setup asks for a company or local proxy URL,
  accepts an optional gateway key, discovers its models and remembers the connection.
- Fresh setup and `/connect` let local runtimes and gateways use their standard or saved
  endpoint, enter another URL, and keep, replace or remove an existing server key. No
  configuration-file editing is required.
- `/effort` uses the selected model's advertised capabilities: unsupported levels are
  unavailable, while unadvertised self-hosted capabilities are clearly marked as an
  explicit try-at-your-own-endpoint choice. Requests omit known-unsupported effort settings.
- Provider-neutral setup now offers each provider's supported connection methods. OpenRouter
  can connect through a PKCE-protected browser flow or a one-time code on remote machines;
  manually pasted keys and environment credentials remain supported.
- Hosted-provider labels include their familiar model families (Claude, GPT and Gemini)
  so users do not need to know the vendor/company distinction.
- `/connect [provider]` adds or replaces a provider connection without leaving the session.
- Setup discovers models from Ollama, LM Studio, llama.cpp, vLLM and SGLang on their
  standard local endpoints, then checks the selected model before starting a session.
  When the standard endpoint is unavailable, it can connect and remember a remote or
  custom endpoint with an optional server key.

### Changed
- `/model` shows the complete model catalog for the active provider instead of a
  mixed-provider recent list. The banner and status bar identify the provider separately
  from its model.

### Fixed
- Ollama setup reads each model's native capabilities and automatically uses text-based
  tool calls when a model such as `gemma3:270m` does not accept API tool definitions.
- GitHub Copilot setup uses the account-specific API endpoint returned by GitHub and lets
  you choose another model without signing in again when GitHub lists one it later rejects.
- GitHub Copilot setup recognizes Free and Student accounts as Auto-only instead of showing
  catalog entries they cannot select. Paid and organization-managed accounts get Auto plus
  the named models GitHub explicitly enables for their account and policy.
- GitHub Copilot's Auto model works. It failed with "The requested model is not supported"
  because tinyharness sent requests without a model; it now asks GitHub which model Auto
  picks and sends that model with Auto's session token, as VS Code does.

## 0.1.0 (2026-10-14)

First release.

Release validation is summarized in
[`docs/releases/v0.1.0-smoke.md`](docs/releases/v0.1.0-smoke.md). Raw session traces are
intentionally excluded because they can contain prompts, filesystem paths and project context.

### Changed
- Published the package on PyPI as `tinyharness-lm`; the `tinyharness` command,
  Python import, and project name stay the same.
- Switched the project license from MIT to Apache License 2.0.

### Fixed
- Native Windows CI now isolates each test's home directory, accepts portable permission
  rules and diagnostics, stops timed-out hook process trees, and does not apply POSIX-only
  terminal or file-mode assertions.
  Sandbox Runtime keeps its Linux control sockets in a short system temp path while each
  sandboxed command still receives its private session temp directory.
- Windows in-process file reads and writes now reject reparse-point swaps after approval;
  directory handles keep every checked parent from being replaced until the operation ends.
- Software-acquisition approval now covers shell conditionals, wrapper options such as
  `env -u` and `env -S`, and versioned `pip` commands. Absolute Python package installers
  are rebound at their parsed invocation—not an earlier textual match—to the writable
  environment named by the approval, while fresh-environment bootstrapping stays separate.
- After outside content enters a conversation, sandboxed Bash asks before running whenever
  its network policy has an outbound destination; an allowlist limits destinations but is
  not treated as consent to transmit project data.
- Windows-style paths now receive the same case-insensitive credential-file protection as
  POSIX paths. Replies stopped at their output limit cannot execute any tool call, even when
  one call looks complete. Software-acquisition approval now recognizes newline-separated
  commands, common Python/package-manager options, and dependency scripts after `cd`.
- Context, output tokens and cost now update with lightweight estimates while a model is
  streaming, then reconcile to the provider's exact usage when each response completes.
  The locally tokenized next-request context remains visibly approximate while exact output
  and cost totals retain their provider-reported values.
- Sandboxed test suites can read credential-shaped fixtures created beneath the session's
  private temporary directory without exposing real credentials. Commands that need a local
  test, preview or development server can request the scoped `localhost` capability while
  remaining sandboxed, and a denied loopback bind now explains that recovery path.
- Git diff previews color additions green and removals red, including `--stat` graphs,
  without accepting terminal control sequences from command output.
- Sequential tool rows now measure execution time rather than time spent queued behind
  another tool, and source output containing timeout-related text no longer creates a
  false timeout.
- Software-acquisition approvals are bound to the complete shell command, including every
  chained install. Combined shell flags such as `bash -lc` and `uv run --with` dependencies
  can no longer bypass the approval boundary.
- Sandbox Runtime now augments its existing home-directory protections by translating the
  shared credential policy across the project and sibling repositories. Real Linux SRT tests
  cover write confinement, project and sibling credentials, and blocked network access.
- Publishing waits for the same lint, type, platform-matrix, minimum-dependency, package,
  and real Linux Sandbox Runtime checks as ordinary CI.

### Added
- The core Open tool now presents files and URLs as typed, clickable artifacts
  inside tinyharness by default. Launching a browser or host application remains a separate,
  explicit mode that always crosses the permission boundary.
- Todo tasks now have stable IDs across updates and resumed sessions, plus blocked and
  cancelled states and a short result field for completion evidence or blockers.
- Memory changes now have a private append-only audit ledger with stable IDs, timestamps,
  source session/turn, outside-content provenance, and `/memory history`. The small Markdown
  snapshot remains the only memory content loaded into model context.
- `/sessions` opens a searchable conversation switcher inside the running CLI. The same
  searchable picker is used for models, providers and skills; choosing a skill prepares its
  slash command so arguments can be added before it runs. Session logs use an OS-held lock,
  preventing two tinyharness windows from writing the same conversation at once.
- Explicit software acquisition through common package managers, tool launchers, bootstrap
  scripts, browser runtimes, model downloaders and container clients now asks before downloading
  or executing new code, even inside the sandbox or `bypassPermissions`. The exact acquisition can be
  allowed for the session; using an already-installed executable does not ask again.
  `uv run` also asks when its target is a PEP 723 script that declares dependencies; ordinary
  installed commands such as `uv run pytest`, `--offline`, and `--no-sync` stay quiet.
  Network-host approval stays separate, while plan mode continues to block acquisition.
- Approved bare Python installs reuse an active writable virtualenv or Conda environment.
  When the launcher environment is outside the sandbox or none exists, tinyharness creates
  and uses `.venv` in the project instead of modifying a global or harness environment.
- Linux and Windows can use Anthropic's optional Sandbox Runtime as the same filesystem
  and network policy backend; native Seatbelt remains preferred on macOS. No new Python
  dependency is installed, and `/sandbox` gives platform-specific setup help when none is active.
- `[sandbox] required = true` now fails closed at session startup when no supported
  sandbox is active, so unattended runs cannot approve an unsandboxed fallback.
- Session JSONL records now use trace schema 1 and carry session, turn, agent, model-call,
  routing-decision and tool-call IDs plus the harness version. Main-loop model calls and
  permission, tool, compaction and total-turn latency are recorded for evaluation.
- Background model policies can now be set independently under `[models.roles]` for the
  safety classifier, search digest and compaction. Lead and subagent compaction use the
  same resolution; provider `small_model` and `compaction_model` remain compatibility fallbacks.
- SDK sessions accept post-turn `evaluators`; each gets a deep-copied trajectory and its
  result, error or timeout is appended to the trace without changing the completed task.
  `examples/09_evaluator.py` shows the complete observer flow.

### Changed
- Resuming no longer requires remembering an opaque session id: `-r` opens a title-based
  picker with recency, message count and model, while `-r words` searches conversation titles.
  Exact ids remain supported for automation.
- A blocked proxy-aware download now returns a structured failure with the exact destination.
  Bash may request that one host; the user can allow it once or for the session while the
  command remains sandboxed. Broad redirect hosts are not trusted by default.
- Shell commands run fail-fast, including pipelines, so a denied or failed intermediate
  step cannot silently continue and consume a stale artifact.
- Bash containment is selected entirely by the harness; its tool schema exposes no
  containment control. Permission modes, allow rules and allow hooks cannot silently
  cross a missing-sandbox or host-app boundary.
- Sandboxed commands now use a per-session temporary directory and a private tinyharness
  cache, including explicit locations for Playwright and other common package/model
  installers, instead of inheriting broad system temporary directories.
- API keys now go through one resolver with visible provenance: SDK override, process
  environment, development `.env`, then `auth.json`. Saved and dotenv keys are no longer
  copied into `os.environ`, and `/whoami` reports the selected source.
- Public SDK and extension examples now use concise numpydoc documentation. Function
  tools read their input descriptions from numpydoc `Parameters` sections while keeping
  the earlier compact form compatible.
- Code comments and documentation now explain tinyharness behavior directly, without
  relying on comparisons to other agent products or their implementations.
- Credential-based provider fallback is now named `credential_selected` internally;
  `auto_selected` remains a compatibility alias and no longer occupies future routing terminology.
- Prompt-suggestion shown/accepted events are recorded without their text so the default
  model call can be evaluated before changing its behavior.
- Removed the obsolete compaction implementation plan; the current design remains in the
  feature documentation and `agent/compaction.py` rationale.

### Fixed
- PEP 723 scripts launched after a shell directory change can no longer bypass the
  software-acquisition approval when their path cannot be inspected before execution.
- Resuming a conversation now restores the optional extensions required by skills already
  loaded into its context, and migrates older saved todo lists to stable task IDs.
- Memory audit entries after `/clear` now carry the new conversation's session ID rather than
  the cleared conversation's ID.
- Exact network destinations approved for a session now accumulate in the sandbox policy,
  not only in the permission prompt state. Multi-host clients such as market-data libraries
  can continue across redirects and service hosts without disabling containment or losing
  earlier approvals; one-command approvals remain limited to that command. An invoked skill
  can declare a finite `requires-network` set, letting the user review and approve all of its
  exact destinations once for the session instead of discovering them through repeated failures.
- Invoked skills can declare `requires-extensions` in their frontmatter. tinyharness turns
  those capabilities on for the session before the first model call, so a multi-agent skill
  receives `Agent`/`Team` instead of silently collapsing isolated roles into the main context.
- Indirect PEP 723 launches such as `uv run $SKILL_DIR/scripts/tool.py` no longer bypass the
  software-acquisition prompt merely because the harness cannot resolve the shell variable.
- Reading an artifact is no longer described to the model as showing it to the user. Requests
  to show or open a result use the Open capability when available, or get an honest fallback.
- Memory writes now require an explicit `user` or `project` scope, and the tool and
  `/memory` explain when a durable fact belongs in memory versus a user `AGENTS.md` or
  shared project `AGENTS.md`.
- Per-project memory, session logs, sidechains and scratch files now live under a readable,
  collision-resistant directory such as `projects/tinyharness-8f31c2a4d6e709ab/`, rather
  than a directory containing the machine's full absolute path. Existing state is migrated
  automatically on first use.
- The distribution smoke test now checks the source version's wheel instead of silently
  selecting a stale, higher-version artifact left in `dist/`.
- Recognizable destructive commands (`rm`, `find -delete`, `git reset --hard`, and `git
  clean -f`) require one-time explicit approval in auto mode while routine sandboxed work
  remains uninterrupted.
- Context compaction now resumes from an execution-state checkpoint instead of repeating
  the original request as the newest user instruction. Completed tool, Agent, and Team
  work is explicitly marked not to repeat, preventing long tasks from restarting after a
  summary while preserving compatibility with older saved sessions.
- Back-to-back permission questions no longer hang when tinyharness runs in a dumb or
  redirected terminal: the input box now releases the actual fallback application.

### CLI
- Interactive coding agent with streamed Markdown, live tool status, `/` command
  autocomplete, pickers for model, provider, effort and permission mode.
- Permission prompts with diff previews; "don't ask again" per project or directory;
  Shift+Tab cycles `default` → `acceptEdits` → `plan`; persistent allow/deny rules.
- Guide on first run when no API key is set. Switching to a provider with no key
  (`/provider`, `/model`) asks for it right there, hidden as you type, checks it with the
  provider and saves it; `/key` adds or replaces one any time. No file editing needed.
- Everything of yours is in one folder, `~/.tinyharness/`: `config.toml`, `auth.json`
  (API keys, one entry per provider, readable only by you), `skills/`, `agents/`,
  `plugins/`, `hooks.toml`, state and session logs. A key set in your shell still wins.
  Earlier versions' `~/.config/tinyharness/config.toml` and `.env` keys are moved there on
  the first run; anything that can't move stays put, with a warning. Any file named
  `auth.json` is secret-shaped: tools can't read it and the sandbox blocks it.
- Tool calls as one aligned row each, colored by kind, with the result on the right
  (`+14 −3`, `212 passed`, `exit 1` plus the error line); subagents nested under their call.
- The input box stays up while the agent works: its progress is drawn above it, and a
  message typed meanwhile reaches the agent before its next step (or starts the next turn,
  if it finished first). Ctrl+C in the box stops the turn. Permission prompts take the
  keyboard and give it back with what you'd typed, and collapse once answered.
- The model's reasoning, where the provider shares it (Anthropic thinking, OpenAI reasoning
  summaries, Gemini thought summaries, OpenRouter, DeepSeek, Mistral): hidden by default as
  `∴ Thought for 8s`; `ctrl+t`, `/thinking` or `show_thinking = true` show it, streamed
  while the model reasons. OpenAI summaries are only requested while it's shown.
  SDK: `Reasoning(text=...)` events and `Session.set_show_thinking()`.
- Turn footer: time, tools, files changed, test result, the largest context any call
  sent, output tokens and cost (`83k context · 9.4k out · $0.06`). Status bar: model,
  sandbox, git branch (✱ when dirty), context use, output tokens and cost; narrow
  terminals drop the least important parts first. Neither adds up every call's input:
  each call re-sends the conversation, so that sum grows with every step and is mostly
  cheap cache reads. `/cost` shows the full breakdown. Output and cost are the
  session's, so `/clear` keeps them (it says so, and `/cost` adds what was spent since).
- Token usage from every provider's stream (`TokenUsage` events; subagents included in
  the session's total). Cost when the provider reports it (OpenRouter), its model list has
  prices, or `[pricing]` in the config sets them; otherwise tokens only, never a guess.
- The input box grows only by the `/` menu entries it shows, not a fixed 6 rows.
- Skills: `/<name>` anywhere in a message loads that skill with it (not only at the
  start), unless it's already in the conversation. The CLI shows each load
  (`⎿ skill x loaded · 9.2k tokens`), including the model reading a SKILL.md itself.
  SDK: `SkillLoaded` events.
- `/btw <question>`: a side question answered from the conversation so far, without
  tools, even while the agent works; neither question nor answer joins the conversation.
  The answer shows above the input box until Esc closes it; `/output` prints it.
  It reuses the agent's cached prefix (one short call). SDK: `Session.aside()`.
- After each turn the empty input box suggests your likely next message (Tab takes it):
  one short call per turn, mostly cache reads. `prompt_suggestions =
  false` turns it off. SDK: `Session.suggest_next()`.
- `ctrl+t` after a turn shows its hidden reasoning once, not again on every press.
- Images: the model can see them. `Read` shows it a PNG, JPEG, GIF or WebP (a
  screenshot it took of a page or a deck, a chart) instead of refusing it; image file
  paths in your message (pasted screenshots, escaped spaces and all) are attached to
  it. Sent as each provider takes them: inside tool results for Anthropic, in a message
  right after them for the others. Shrunk to fit (see below). SDK: `session.run(prompt, images)`,
  `tools.read.load_image`, and `ToolOutput` for tools that return pictures.
- Checking a page by looking: the model renders it with a headless browser through
  sandboxed Bash and Reads the PNG. Browser and package caches remain private; redirected
  download hosts need exact, visible approval. Desktop Chrome remains a host application
  and is not a sandbox workaround.
- A running subagent shows its last few tool calls under it (outcome, first line of the
  result, time) with a count of earlier ones, so you can tell progress from going in
  circles; `ctrl+o` while one runs shows all its calls so far. Pasting a file path at
  the start of a message no longer runs it as a `/command`, and `Read` refuses binary
  files instead of filling the context with their bytes. SDK: `SubagentUpdate.recent`
  (`SubagentCall`s) and `.log` (its transcript).
- A parallel batch shows each call's result as it finishes, with its own time: a quick
  Read no longer waits for a slow WebSearch and shows the batch's time. The model still
  gets the results in the order it asked for them. SDK: `ToolUse` and `ToolResult` carry
  the call's `id`, and a parallel batch's `ToolResult`s come in the order they finish.
- Stream events that aren't valid JSON are still skipped, but now with a warning (the
  reply may be missing text), not silently.
- While the model writes a tool call (a whole file can take minutes), the spinner says
  so, `Writing out/deck.html · 18.4k chars…`, instead of "Thinking", and the thought's
  time stops when the writing starts. SDK: `ToolInputStreaming` events (the tool, how
  much has arrived, its path, URL or command), about one per KB.
- A running subagent's row says what it's doing (`Thinking`, `Working`, `Writing
  deck.html`, or the tool it runs), and its clock keeps ticking between updates, so a long
  model call no longer looks frozen at the time of its last tool. SDK: `SubagentUpdate`
  has `phase` and `tool`.
- The Bash tool tells the model that, in the sandbox, package managers can't download
  and other apps (`open`, `osascript`, browsers) can't be reached, so it stops probing
  for them: `osascript` hangs until the timeout rather than failing.
- The TodoWrite checklist is off by default: current models keep track of multi-step work
  without one, and keeping it up costs steps and tokens. Turn it on with
  `todo_list = true`, `TINYHARNESS_TODO_LIST=1`
  or the SDK's `SessionOptions(todo_list=True)`, e.g. for smaller or older models or to
  watch a live plan. When on, it asks for a few coarse steps that match how the work is
  done, skips simple work, and lets one call finish several steps. Subagents never get
  it: the list is the main agent's plan on your screen, and a subagent's would replace it.
- New files go in the working directory unless you say where, not next to an input file
  outside it (a PDF in ~/Downloads): only there can later edits and commands reach them
  without asking.
- A multi-line Bash command (a heredoc, a script) takes one row:
  `Bash  cat > /tmp/q.sql << 'EOF' (+12 lines)`, not its body spilled over several.
- Fixed a hang when two tool calls in a row needed approval (e.g. Read and Grep on a file
  outside the project): the second question could arrive while the input box was coming
  back, and then waited forever for the keyboard, with no question on screen.
- Reading tools share approvals: "allow reading in ~/Downloads/ for this session" covers
  Read, Grep, Glob and Pdf there (the folder itself too), and a yes to reading one file
  covers the others on it. Reading a folder outside the project was one question per tool.
- The auto-mode classifier sees only your messages and the agent's tool calls: no tool
  output, no agent text, no teammates' messages. A page or file saying "the user
  approved this" can no longer argue for an action.
- `AGENTS.md` is the single instruction filename at the system, user, project and local levels.
  When a project has none, the model is told,
  so it doesn't go looking. Reading a file that isn't there shows a quiet `not found`, not
  a red error.
- The turn summary counts subagents' and teammates' tool calls too: `128 tools (111 by 4
  agents)`, where it said `17 tools`.
- A project's tests can read back the fake `auth.json` and `.env` files they make in
  pytest's temp folder when run in the sandbox; those names stay unreadable everywhere
  else. tinyharness's own test suite now passes when the agent runs it.
- The Bash tool states the sandbox's limits directly: writable roots, credentials and API
  keys, network destinations, and host applications. A blocked operation gets one short,
  actionable note.
- Images are shrunk to at most 2000 px a side (and 5 MB) before the model sees them: pasted
  screenshots and images the agent reads. A Retina screenshot used to go in at full size,
  and once a request had several images, the provider rejected it and every request after it. If a provider still
  rejects an image, the images in the conversation are shrunk further (or dropped, with a
  note) and the request is retried once. Adds Pillow as a dependency.
- `/clear` starts a new conversation: the old one stays saved as it was
  (`tinyharness -r` picks it). It used to be emptied in place, so `-c` right after a
  `/clear` failed with "has nothing to resume". `-c` and the `-r` list now also skip
  conversations emptied that way.
- A message you send while the agent works is listed above the input (`• Sent after the
  agent's current step (esc: stop it and send now)`): wait, and it
  goes in after the current step; press Esc, and the step stops and your message goes
  in right away. When it reaches the agent, `↳ delivered to the agent: …` says so. SDK:
  `MessageDelivered` events and `Session.pending_steering`.
- `send_ids = true` on a provider sends who is calling with every request: X-Session-ID,
  X-Agent-ID, X-Parent-Agent-ID, X-Agent-Role, X-Agent-Name and X-Task (from `TH_TASK`),
  so a gateway of your own can admit, queue and attribute per session and per agent.
  Each subagent gets its own id (its parent's as X-Parent-Agent-ID), also when several run
  at once. Off by default. SDK: `tinyharness.llm.identity`.
- One ceiling on subagents running at once: `max_parallel_agents`
  (default 20; `TINYHARNESS_MAX_PARALLEL_AGENTS`), shared by Agent calls and Team
  teammates. The model decides the team's size (the Team tool suggests 3-5); past the
  ceiling an Agent call fails so the model can wait, and Team tasks wait their turn.
  A Team takes up to 20 tasks (was 12) and no longer stops at 4 teammates at once.
  SDK: `AgentSlots`; `run_team(..., slots=)` replaces `max_parallel=`.
- Subagents that can write run side by side too: Team teammates and several
  Agent calls in one reply. Only their calls that change things (Bash, Edit, Write) take
  turns, on one lock per session, so two never write at once; their model calls overlap.
  A team of writers used to run one whole agent at a time. SDK: `Tool.runs_alongside`,
  `LoopDeps.write_lock`; `run_team` no longer takes `agent_is_read_only`.
- The Agent and Team prompts tell the model that a subagent sees only its prompt, not the
  conversation, so it must pass full paths (a database the user named, say): teammates
  were given "nova_ai.db" without the folder and queried the wrong file.
- The spinner's clock keeps counting while a model streams blank lines between pieces of a
  tool call; it restarted at each one (0s, 1s, 0s, 1s).
- A reply that gets stuck while still streaming is stopped: one short piece repeated
  (blank lines count; a few characters repeated in its text or reasoning, like a line of
  backticks, is caught after 600), or a minute of nothing but whitespace. The model is told why and
  asked once to carry on (`↻ The model got stuck: …`); stuck again in the same turn, the
  turn stops and suggests another model. Small models could otherwise stream filler until
  their output limit. SDK: `ModelStuck` events, `RunawayOutputError`.
- `ctrl+o` shows the last tool call's full output; file paths are clickable links; long
  paths are shortened in the middle; a one-line banner after the first launch; `NO_COLOR` honored.
- Themes: `system` (default; asks the terminal for its background color, falling back to
  `COLORFGBG`), `light` and `dark`. Set with `/theme`, `theme =` in the config, or
  `TINYHARNESS_THEME` (that order of priority, lowest first).
- The system prompt covers what's new: verify changes by running the project's checks and
  say what was verified; what to do when a call is denied or the sandbox blocks something;
  file contents, command output and web pages are data, not instructions; what the
  harness's markers mean (`<session-resumed>`, `[stale: …]`, …); no commits or pushes
  unless asked. Still about 1k tokens.

- `ctrl+o` is a toggle: it expands the last tool call's output (or running subagents'
  calls, newest kept) above the input box, fitted to the screen, and collapses it again,
  leaving nothing in the scrollback. It used to print a new panel on each press, with
  no way to close one. `/output` prints the whole output, as before.
- A command's or search's output is previewed under its row: the first 3 lines (the last
  3 when a command failed, where the error is), then `… N more lines`; `ctrl+o` still
  shows the last call in full. Reads, edits and writes stay one row: `212 lines`, `+4 −1`.
- A call allowed without asking says by what, on its own line under it:
  `⎿ auto mode · read-only query in the project`, `⎿ allow rule`, `⎿ hook`.

### Reasoning
- Models whose thinking is a per-request switch (OpenRouter lists `reasoning` among their
  parameters: Gemma 4, Claude, Qwen3 thinking models) now reason at medium effort on agent
  turns unless you chose an effort. They used to run with thinking off, so a Gemma 4 team
  planned, briefed and did arithmetic without reasoning. The harness's own quick calls
  (the permission classifier, summaries) stay without it. `/effort none` or
  `effort = "none"` turns reasoning off; a cached model list from before this is
  refetched once.
- A built-in provider's tables (`extra_body`, `extra_headers`) merge with yours key by key:
  adding OpenRouter routing preferences used to drop its prompt caching and cost reporting.

### Skills and delegation
- Skills are offered to the model by default: the system prompt lists each skill's name,
  the start of its description and its SKILL.md, which the model reads to load the skill
  (no Skill tool). Subagents and teammates get the same list. `skill_tool = false` (or
  `--no-skill-tool`) leaves skills to `/<name>` only.
- A teammate's result longer than a prompt carries (4,000 characters to a dependent task,
  6,000 in the lead's report) is saved whole in the session's scratch folder, and the next
  task and the lead are told where: it used to be cut off with no way to get the rest, so
  a report builder saw only the start of each analyst's findings.
- A skill you name in your message ("use the claude-design-skill for the report") isn't
  optional: the system prompt says so. (A note appended to every message naming a skill,
  even "don't use it", was dropped as a duplicate.)
- The system prompt and the Agent and Team tools tell the lead to brief subagents like a
  capable colleague new to the work (what it's for, how deep to go, what to deliver),
  and to ask for exact data, not only prose, when one agent's output feeds another.

### Working habits
- The system prompt adds habits that hold for any coding task: reproduce a bug before
  fixing it, fix the cause and re-run the same steps, never silence an error; after two
  failures on the same step, rethink instead of trying a third variation; try what you
  built the way a user would, awkward cases included; no unasked-for dependencies or
  renames; and give each subagent one job with a clear done condition.

### Choosing a provider
- With no provider or model named, a key set for this run (your shell, a `.env`, an IDE's
  env file) wins over one only saved in `auth.json`: a stale saved Anthropic key used to
  be picked over a fresh OpenRouter key, because Anthropic is first in the list.
- `examples/06_subagent_models.py` uses the session's model for the second opinion unless
  you pass another provider's model, so it runs on any one key.

### Tool-call format per model
- How a model calls tools is measured once, not guessed: the first request to a new model
  offers it one tool. Native callers keep the API's tool interface; a model that writes
  calls in its reply gets the tools described in its prompt and writes
  `<call>{"name": ..., "arguments": ...}</call>`, with earlier calls and results sent as
  text. Cached per model; `tool_calls = "native" | "text"` overrides it. A server without
  a tool-call parser (vLLM without `--enable-auto-tool-choice`) works this way.

### Outside content (Rule of Two)
- Once content from another machine is in the conversation (a fetched page, search
  results, a command's download through the sandbox proxy), calls that could send data out
  ask a person, in every mode but bypass: network tools no longer read public pages
  without asking, auto mode's classifier doesn't decide them, and leaving the sandbox says
  outside content is present. Sites you approved stay allowed; sandboxed commands are
  unaffected; `/clear` starts over; `/sandbox` shows the sources. Subagents share it.

### Extension needs
- An extension's `NEEDS` (`network:<host>`, `write:<path>`) are enforced. The CLI asks
  once when it loads (remembered with a fingerprint of its code; a change asks again);
  approved, its own tools reach those hosts and folders without asking, and
  `th.run(command)` runs commands in the sandbox with them added. Declined, it loads
  without them. `/extensions allow <name>` approves later.

### Permission modes
- `default` no longer asks for routine work: reads, edits inside the project and
  sandboxed commands run; operations in another folder and network tools ask. The new
  `ask` mode asks before every edit and
  command, as `default` used to. `acceptEdits` still works and means `default`.
  Shift+Tab cycles `default` → `auto` → `plan` → `ask`. Reading a public web page (a URL
  without a query string) doesn't ask in `default` either, as in `auto`.

### Sandbox network
- Sandboxed commands can reach an allowlist of hosts through a small local proxy
  (`sandbox/proxy.py`): PyPI, npm, GitHub, the common script and font CDNs, and
  Playwright's browser downloads. `pip install`, `npm`, `git clone` and a page that loads
  Chart.js from a CDN (rendered with `npx playwright screenshot --proxy-server
  "$HTTPS_PROXY"`) work without asking. Seatbelt lets commands connect only to the
  proxy; other hosts get a 403, and the command's output names the blocked host. Change
  the list with `[sandbox] allow_hosts`; `[]` turns the network off. `/sandbox` shows
  what commands can write, read and reach.

### Extensions
- Extensions: one Python file with `setup(th)` adds tools (from an async function's
  signature and docstring), prompt sections, slash commands, and handlers for
  `user_prompt`, `tool_call`, `tool_result` and `turn_end` (the same bus as `hooks.toml`).
  Turned on in `[extensions] enabled = [...]` or with `/extensions on|off`, which lists
  each one's tokens per request. See docs/extension-api.md.
- The core is seven tools: Read, Write, Edit, Glob, Grep, Bash and Open. Web, Pdf and TodoWrite
  are built-in extensions (`web`, `pdf`, `todo`), off until turned on, so a session
  that doesn't use them doesn't send their schemas with every request.
- Memory across sessions, the `memory` extension (off until turned on): the agent keeps
  short notes about you (`~/.tinyharness/memory.md`, every project) and about the project
  (`~/.tinyharness/projects/<project>/memory.md`, never in the repository) with a Memory
  tool, and every session starts with them in its prompt. Notes saved during a session
  show from the next one, so the cached prompt stays the same. Each memory has a small
  size limit (1,500 and 3,000 characters); when one is full the agent merges or drops
  notes. Saving needs no prompt, except while content from the web is in the
  conversation (it could be planting a note) or in `ask` mode. Notes that look like API
  keys are refused. `/memory` shows both files; they're plain Markdown you can edit.
- Agent and Team are the `agents` extension, built only on the public API
  (`th.run_agent`), with their briefing guidance as its prompt section.
- The core system prompt is about 670 tokens (was about 2,300): how to work, safety, pages,
  harness notes and answering. With no extensions on, the whole prefix is about 2.9k tokens
  before your AGENTS.md and skills (was about 7k).

### Reports and pages
- Pages are written as models write them elsewhere, CDN links included: no offline
  rule, bundled Chart.js or Screenshot tool. Those sent models hunting for a library file
  (`find /` for two minutes) or downloading one, and Write rewrote the page behind them.
- Prompts carry no one demo's rules: the business-database Team example, "delegate
  without being asked", "re-check a subagent's numbers" and "never do mental math"
  (system prompt), and "never make up data" and "compute every number with code"
  (teammates) were dropped. They pushed models into costly teams and double work.

### Project instructions
- AGENTS.md files are read from the repository's root down to the
  folder tinyharness runs in: launching in `src/` still gets the repo's
  rules, and a subfolder's file refines them. Outside a git repository only the working
  directory counts, so a file in a parent folder doesn't apply everywhere.

### Look and feel
- A new identity, "Field Notes": ink blue on paper, with a highlighter for what matters.
  The CLI's palette follows it in both themes, and the wordmark is "tiny" under the
  highlighter.
- Meet Scout, the tinyharness pony. He's on the README's notebook-page banner, and in the
  terminal he greets you beside the session details as pixel art (only in color
  terminals wide enough for him).
- Tool rows lead from the target to the result with dot leaders; the turn summary starts with a rule; a passing test run shows
  its result on its row (`3 passed`) instead of output under it.
- The README is rewritten around what tinyharness is and how to start, with a real
  session screenshot; the full feature reference and configuration guide moved to
  `docs/features.md` and `docs/configuration.md`.

### Reliability
From an audit of 157 real sessions across 9 models and the code. Each item has a
regression test.
- Every reply has one of a shared set of outcomes (`stop`, `tool_calls`, `length`,
  `filtered`, `refused`, `paused`), normalized from each API's own words. A reply cut off
  at the output limit is continued instead of taken as the answer; a tool call cut off
  mid-arguments isn't run, and the model is told to write in smaller pieces.
- The history sent to a provider is always well formed: every tool call answered right
  after it, empty assistant messages dropped, and tool-call ids another provider wouldn't
  accept replaced (deterministically, so the prompt cache still matches). Ctrl+C keeps the
  results of the calls that did finish.
- Context use starts from the provider's own token count, and the estimate counts
  tool-call arguments and images (a Write's content used to count as nothing). A summary
  can cut inside one long task, repeats the request word for word, and sees the tool
  calls, files changed, todo list and any earlier summary in full. Old tool calls' long
  arguments are trimmed like old results.
- Tool calls written as text (JSON and XML forms, including model-specific variants) are
  recovered when they name a real tool (Gemma's raw `<|"|>` strings included, multi-line
  commands with quotes and all); a reply ending in a call that can't be used (a made-up
  tool, unreadable arguments, or a call's bare JSON with no tags) is answered with what went wrong instead of ending the turn; malformed arguments are repaired (a code fence, double encoding, two
  objects run together); a number or object for a text field is taken as text; a wrong
  case in a tool name still finds the tool; invalid arguments are echoed back.
- Streams: busy errors and connection resets before any output are retried;
  `retry-after-ms` is honoured; multi-line SSE events are joined; a body that isn't an
  event stream is reported; parallel calls from servers without `index` stay separate.
- Grep anchors work in every mode; Glob skips dependency folders; Edit keeps CRLF files
  CRLF, refuses an empty `old_string`, drops Read's line numbers from `old_string`, and its
  whitespace-tolerant match no longer adds blank lines; Bash runs non-interactively, its
  timeout covers the whole command, and omitted output is saved to a file it names; Read
  pages by size and always says how to continue.
- The system prompt states the OS (BSD tools on macOS), shell, Python command and git
  branch. Subagents get these and the project's AGENTS.md.
- Finishing: the turn ends when the model does (the system prompt asks it to verify its
  work; an extra "you haven't run anything" round-trip after every edit was dropped), and
  the main session's step limit is 200 (was 50). A Stop hook can hold the turn open
  (`command = "uv run pytest -q -x"`), and a PostToolUse hook's reason reaches the model.
- Teams: a teammate's answer is its last reply as kept; TeamBoard waits for news instead
  of being polled; a failed subagent hands back its last calls and words.
- The session log records how each turn ended (done, interrupted, or the error), the
  provider's stop reason, and OpenRouter's upstream route.
- Earlier copies of a file are no longer rewritten as stale on every edit (that broke
  prompt-cache reuse); the edit's snippet and Edit's stale-file check cover it. Resume
  still marks changed files stale, and now remembers file versions so a file changed
  after resuming must be read again.

### Resume
- Conversations are saved losslessly as they go (full tool results, reasoning state,
  compactions, todos, model) and resumed with `-c` (latest) or `-r` (picker), or
  `SessionOptions(resume=...)`; `list_sessions()` lists them. The session log lives
  under `.tinyharness/`, which sandboxed commands can't write to.
- On resume the harness checks what changed while the session was away: files the
  agent read (content hashes), git HEAD or branch, and AGENTS.md. Old contents of
  changed files are replaced by a stub, the agent must read them again before editing,
  the model is told once what changed, and tool calls with no recorded result are
  flagged as maybe run. The system prompt is rebuilt; the conversation's model is kept
  when there's still a key for it.
- `SessionStore` interface (`FileSessionStore` by default) for keeping conversations elsewhere.
- tinyharness never writes into your project: conversation logs and scratch space live
  in `~/.tinyharness/projects/<project>/`, so they can't be committed. Logs from earlier
  versions are moved there, and an emptied `<project>/.tinyharness/` is removed.

### Providers
- Native APIs: Anthropic Messages, OpenAI Responses (OpenAI, xAI), Gemini
  Interactions, and Chat Completions (OpenRouter, DeepSeek, Mistral, Groq, gateways).
- Reasoning carries across tool calls; one `effort` setting for every model. That
  includes providers that stream it only as `reasoning` (Groq, vLLM, Ollama).
- Models, context windows and capabilities come from each provider's model list.
- Models are checked before switching; context overflows are compacted and retried.
- Old tool output is cleared in batches, not one result per turn, so the prompt cache
  survives. While the cache is warm, output is only cleared once the history passes 75%
  of the room before a summary, and only when that frees enough (15% of it, at most 20k
  tokens). Results under 3.6% of it (at most 6k tokens) are left alone. On a 64k window
  that clears 1-2k-token results before a summary is needed; before, it never did. After 5 idle minutes, when the cache has expired anyway, it's cleared freely.
  The most recent tool output is kept (30% of the usable window, at most 40k tokens), and
  so are skill reads, `Agent` and `Team` results.
- Compaction sizes its limits from each model's window, so a 64k model gets them too: the
  protected tool output above, and the largest single message (half the usable window,
  at most 16k tokens). An oversized message keeps its start and its end (where a test
  run's result is), and trimmed output is saved to a file in the session's scratch space,
  named in its place, for the model to Read back instead of running the tool again.
- Requests ask for the output room compaction keeps for the reply (the model's output
  limit, at most 32k tokens or half the window) when the model list or config gives that
  limit; summaries ask for at most 8,192 tokens. After a context overflow, the summary
  keeps only the last ~2k tokens word for word, so the retry fits.
- Without `compaction_model`, a subagent's summaries are written on its own model; one on
  another model of the session's provider used the session's model.
- A subagent's transcript (`sidechains/`) records the session, agent id and parent that started
  it, so it can be matched to its session when several run at once.
- `auto` mode's permission classifier sends its calls as role `classifier` (`X-Agent-Role`), as
  summaries are `compaction`, so a gateway can tell them from the agent's turns.
- A Team saves every member's result to a file and names it, short ones too, so a lead
  never goes looking for a file that isn't there.
- A model's tokenizer (`tokenizer` in a `models` entry) is read from the local Hugging Face
  cache once downloaded: no request to the Hub, and no warning about unauthenticated
  requests, on every session.
- WebSearch runs on the provider's `small_model`, as the auto-mode classifier already
  did. If the small model can't take a request (no search tool, rate limited), the main
  model answers instead. Compaction summaries stay on the main model: a weak summary
  loses state for the rest of the session.
- A reply whose stream ends without the provider's finish event (a dropped connection)
  is an error, not a finished answer, and isn't added to the conversation.
- Network failures say what happened, to which host, with a code: `can't connect to
  openrouter.ai: connection reset by peer (ECONNRESET)`, not a bare `ConnectError`. The
  code is the OS error (`ECONNREFUSED`), DNS error (`EAI_NONAME`), TLS reason
  (`CERTIFICATE_VERIFY_FAILED`) or kind of timeout. Same for `/ping` and `WebFetch`. SDK:
  `ProviderConnectionError` (an `LLMError`) with `.host` and `.code`.

- Self-hosted servers (vLLM, Ollama, SGLang, llama.cpp): a provider without
  `api_key_env` needs no key and sends no `Authorization` header; vLLM's `max_model_len`
  is read as the context window, so compaction triggers at the server's real limit.
- Requests answered 429, 408/409/425 or 5xx (including 529), and refused connections,
  are retried up to `request_retries` times (default 3) before the reply starts,
  waiting for the server's `Retry-After` when it gives one, otherwise backing off
  exponentially with jitter.
- A reply that goes silent (no bytes for 5 minutes) ends with an error instead of waiting
  out the whole `request_timeout`; one stalled teammate held up its team.
- Tool calls streamed without an id (some OpenAI-compatible servers) get one, so their
  results still match.

### Agents and teams
- An empty reply (no text, no tool call) no longer ends the turn as if finished: the
  model is asked to carry on, and the turn stops only if it happens twice in a row.
- A text reply cut off at the output limit (`finish_reason: length`) is continued.
- At the step limit, the agent gets one last call to report what it found and what's
  left, instead of stopping silently; a subagent that ran out of steps returned nothing.
- Subagents and teammates get an environment block (working directory, scratch
  directory, today's date), and the main system prompt has today's date: without it a
  teammate guessed paths (once querying an empty file with the right name) and
  misread "last month".
- Teammates also see the user's request the team is working on, are told to give files
  names unique to their task (teammates overwrote each other's results) and never to make
  up data that's missing.
- A tool failing the same way twice in a row (same tool, same kind of error) gets a note
  telling the model not to repeat the call and what to try instead. The turn isn't
  stopped for it (four unrelated failures used to end it); the step limit bounds a loop.
  Denials and rejections don't count. A model had made 11 failing edits in a row.
- When a file changes (the agent edits or writes it, or a new Read shows it changed),
  earlier copies of it in the conversation become `[stale: …]` stubs, as they already did
  on resume: models kept copying old text from a Read taken before their own edit.
  Unchanged copies are kept, so the prompt cache isn't spent for nothing.
- Edit and Write refuse a file that changed since the agent last read or wrote it (a
  command, another agent, the user), asking it to read the file again instead of editing
  from memory or overwriting a change it never saw. Each subagent and teammate keeps its
  own record of the files it has read: they used to share the lead's, so one teammate's
  edit counted as seen by all of them, and a teammate could edit a file only the lead had
  read.
- Edit's result shows the changed lines with three lines around them, numbered like Read,
  so a next edit nearby can copy from the file as it is now without reading it all again.
- Edit matches whole lines ignoring trailing whitespace when there's no exact match (a
  blank line written as spaces, a dropped trailing space; indentation still has to
  match), and says so in its result. When nothing matches, the error says where the
  closest copy stops matching and shows both lines, instead of only "old_string not
  found": a model spent 11 attempts on one edit, failing on an invisible blank line.
- `Team(context=...)`: a shared brief (a schema already read, definitions, conventions)
  that every teammate gets word for word, before its own task, so teammates don't each
  rediscover it or guess different definitions. Shared parts of a teammate's prompt come
  first and identical, its own task last, so a server's prefix cache computes the shared
  part once. Capped at 16k characters.
- A teammate that fails (stuck, a provider error) is started once more, told why. A task
  whose dependencies partly failed still runs with what finished, told what's missing;
  it's skipped only when none of them finished (one stuck analyst used to sink the
  team's summary). A failure is written to the teammate's log, and a stopped reply's last
  characters are kept, to tell a real loop from a false alarm.
- `Team` runs any plan that's well formed (known agents, no cycles). It used to refuse
  a plan where no two tasks run at once, and one naming a path whose folder a task was
  still to make.
- Subagents default to 40 steps (was 25); `max_steps` in an agent's frontmatter or
  `AgentSpec` changes it.
- Read-only shell commands (`ls`, `cat`, `grep`, `git log`, `sqlite3 -readonly`, ...,
  joined only by pipes) no longer wait on the lock that makes teammates' writes take
  turns. Anything else, or anything unsure, still does.
- The system prompt says subagents see only the prompt they're given, and that saying
  "I'll delegate" means calling Agent or Team (a small model announced a team, then did
  every part itself).

### Skills
- `--import-skills --force` replaces a skill only once its new copy is in place; if the
  copy fails, the old skill is kept.

### SDK
- `Session`, `query` and a typed event stream shared with the CLI.
- Custom tools, hooks, skills, subagents and plugins.
- Subagents defined in code: `SessionOptions(agents=[AgentSpec(...)])`. `AgentSpec` and
  `SubagentUpdate` are exported from `tinyharness`.
- A subagent's `model` can name any provider you have a key for (`provider:model`),
  or be `small` (the provider's small model) or `inherit` (the default).
- The CLI shows each running subagent's model; `SubagentUpdate.model` carries it for the SDK.
- `SessionOptions(tools=[...])` adds your own tools, which subagents can use too.
- **Breaking:** an agent that doesn't list `tools` now gets every tool the session has
  (except Agent and Team), not just Read, Write and Edit. `tools: []` means none.
- `PermissionRequest.warning`; `remember_label` is empty when "don't ask again" isn't offered.
- `Session` is a context manager (`with` / `async with`), closed however the block ends.
  `ToolOutput` and `ModelStuck` are exported from `tinyharness`.
- `Session.steer(text)`: a message for the agent while a turn runs, delivered before its
  next step; `take_steering()` returns any that arrived too late. `Session.usage` for totals.

### Security
- API keys never reach the model: secret files are unreadable by tools, configured
  key values are masked in tool results, and keys pasted into the chat are caught.
- `Bash` tool, sandboxed with Seatbelt on macOS: writes only to the project, scratch
  and temp dirs; no reads of credential files; no network (`[sandbox] allow_localhost`
  for loopback); no `open`/AppleScript; `.git/hooks`, `.git/config` and `.tinyharness/`
  are read-only. Package managers get their own cache. Where no sandbox is available,
  every command asks. `[sandbox]` in the config adds writable dirs or turns it off.
- Bash containment is harness-owned. Dependency hosts and additional writable roots are
  explicit configuration; host application actions use structured tools and ask.
- The `general-purpose` subagent can run commands (`Bash`).
- Subagents run in the session's permission mode and follow it live: switching to
  auto mid-run reaches subagents already running (so does "allow all edits" at a
  prompt). Breaking: an agent's `permission_mode` is now a limit, never more permissive
  than the session (it used to be the agent's own mode; only plan and bypass carried
  over). Built-in agents have none. A subagent's permission prompt names it, and mode
  changes are recorded in the session log.
- `Agent`'s `subagent_type` is optional (default `general-purpose`): models often left it
  out, and the call failed. Arguments that don't fit a tool are explained briefly (which
  field, what's wrong, what's required) instead of pydantic's full message.
- A sandboxed command that failed with "Operation not permitted" could show as passing
  (`tests ✓` for a failing test run): the sandbox note hid its exit code. The outcome
  now always comes last.
- Read, Pdf, Write and Edit act on exactly the path the permission check approved: they
  don't resolve it again, and open it without following symlinks, so a symlink swapped
  in after the check makes the call fail instead of reaching somewhere else. Writes are
  atomic (a temporary file renamed into place, permissions kept): a crash or full disk
  leaves the old file intact, and directories made for a failed write are removed.
- `Open` tool: shows a document (HTML, PDF, image...) or web page in the user's own
  app; never programs, and it always asks: no allow rule, mode or "don't ask again"
  approves it, since a page in the browser can reach the network. Deny rules apply to
  its path (`Open(private/*)`) or site (`Open(domain:example.com)`).
- A command or hook that times out is stopped along with every process it started,
  including ones that ignore SIGTERM; they get the grace period to exit cleanly first.
- Session logs are private (0600, in a 0700 directory) and credentials are masked in
  them: configured key values and anything shaped like an API key. Shell hooks get the
  same masking in the JSON they receive.
- `.env.example`, `.env.sample`, `.env.template` and `.env.dist` are readable.
- Folder trust: a project's own hooks and plugins (which run outside the sandbox) load
  only once you trust the folder. Only projects that have any ask, once; again only if
  they change. "No" carries on without them. Answers live in `~/.tinyharness/trusted.json`;
  SDK: `SessionOptions(trust_project=...)`. "Change" means any byte of any plugin file, or
  of a project file a hook command names (`./scripts/check.sh`). A project with more than
  50 MB or 5,000 files of these can't be trusted (it says why, and doesn't ask), so it
  can't slow down startup.
- `plugin.toml` paths (`hooks_file`, `skills_dir`, `agents_dir`) must stay inside the
  plugin. A plugin or `hooks.toml` that doesn't parse is skipped with a warning instead of
  stopping the session.
- `WebFetch` connects to the address it checked, not to a second DNS lookup, so a name
  can't be re-pointed at your machine or network in between (TLS still verifies the
  site's name). A redirect to another site isn't followed; the model is told where it
  leads, and fetching it asks about that site.
- `Grep` doesn't follow symlinks, to files or directories (like ripgrep): the
  secret-file check saw a link's own name, so `leak -> ~/.aws/credentials` was read.
  Each file is read once, without following symlinks, so the file checked is the file
  searched.
- `Grep` on a single named file applies the same secret, binary and size checks as a
  directory search.
- Writing `.tinyharness/`, `.git/hooks`, `.git/config` or your user-level settings with
  Write or Edit always asks (except in `bypassPermissions`), since they run code later.
- Agents that come with a project (its files or plugins) can't use `bypassPermissions`;
  they get `acceptEdits`. Yours (`~/.tinyharness/agents`, SDK) are used as written.
