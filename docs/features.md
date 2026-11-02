# Features

Everything tinyharness does, in more detail than the README. For how the pieces fit together, see [How it works](../README.md#how-it-works).

## Reliability

Small and self-hosted models fail in ways the big ones rarely do, and providers fail in
ways no model can fix. tinyharness handles both in the harness, not with model-specific
special cases (see [AGENTS.md](../AGENTS.md#guarantees-dont-break-these) for the
guarantees and why each exists):

- How a model calls tools is measured, not guessed from its name: the first time a model
  is used, one small request offers it a tool. Models that call it through the API keep
  doing so; models that write the call in their reply get the tools in their prompt and
  write `<call>{...}</call>` (a tag no server-side parser strips). Cached per model in
  `~/.tinyharness/cache/tool-calls.json`; `tool_calls = "native" | "text"` overrides it.
- Every reply has a clear outcome, normalized from each API's own stop reasons. A reply
  cut off at the output limit is continued; a tool call cut off mid-arguments isn't run,
  and the model is told to write in smaller pieces.
- The history sent to a provider is always well formed: every tool call answered,
  empty messages dropped, other providers' tool-call ids replaced after `/model`.
- Tool calls written as text (JSON or XML forms) are recovered when a
  server's tool-call parser fails on them; malformed arguments are repaired.
- Compaction starts from the provider's own token count and can summarize inside one
  long task; the summary keeps the tool calls, files changed and the todo list.
- When to summarize comes from the model: once the history leaves less
  room than the model's output limit (at most 32,000 tokens, at most half the window).
  The recent tail kept word for word is a quarter of that usable room (2,000–15,000
  tokens), whole turns first. `compaction_reserved` and
  `compaction_preserve_recent_tokens` override them; `auto_compact_threshold` (a fraction
  of the window) still works.
- A model repeating a failing call is told to change approach.
- The session log records each turn's outcome, the provider's stop reason and, on
  OpenRouter, which upstream served the request.

## Agent loop

Streaming, parallel dispatch of read-only tools, argument validation, Ctrl-C to interrupt a turn. Recovers from what small and self-hosted models do: empty replies and replies cut off at the output limit are continued, a runaway reply is stopped, and at the step limit the agent sums up instead of stopping mid-task. Requests that hit a rate limit or an overloaded server (429/5xx) are retried, honoring `Retry-After`

## Tools

Core: `Read` `Write` `Edit` `Glob` `Grep` `Bash` `Open` `WebFetch`, `WebSearch` and, in plan mode only, `Plan` (the plan in Markdown, in a file outside the project; shown to you in full for review, named in the footer, then one question asks whether to build it in auto or default mode, or keep planning; edit the file to change it) (`Open` provides a safe clickable preview by default, or an explicitly approved external launch; `WebSearch` is offered only where the provider has built-in search: Anthropic, OpenAI, Gemini, xAI, OpenRouter; `[permissions] deny = ["WebFetch", "WebSearch"]` turns the web off, and a tool denied outright isn't offered to the model at all); each one is a class plus a pydantic input model. The rest are built-in extensions, off until you turn them on (`[extensions] enabled = [...]` in config, or `/extensions on <name>`): `todo` (`TodoWrite`: stable resumable tasks with progress, blockers and results; `todo_list = true` also turns it on), `agents` (`Agent`, `Team`), `memory` (`Memory`: explicitly scoped notes about you across projects or about this project that every later session starts with; `/memory` shows both and `/memory history` audits changes). User-authored global instructions remain in `~/.tinyharness/AGENTS.md`, while shared project rules belong in the project's `AGENTS.md`. Your own tools are extensions too: [extension-api.md](extension-api.md). `Read` reads PDFs too, as their text page by page (only the pages it needs), and shows the model images, and image paths in your message (pasted screenshots) are attached to it. To inspect a page it made for visual defects, the model renders it with Playwright through Bash and Reads the PNG; to present the result to you, it uses Open.

## Permissions

In the `default` mode, routine work doesn't ask: reads, edits inside the project and sandboxed commands run. Files in another folder and network tools ask, with a diff preview for edits. Answers are Yes / Yes, don't ask again / No + feedback. Shift+Tab cycles modes (`default` → `auto` → `plan`); `/mode ask` switches to `ask`. Recognizable destructive commands such as `rm`, `find -delete`, `git reset --hard`, and `git clean -f` always ask; an LLM classifier decides other unclear cases. `plan` is read-only: the model is told it's planning (again after the conversation is summarized), researches facts the project or web can answer, and asks you about intent, constraints and meaningful alternatives. When several current datasets, models or approaches fit, it presents researched candidates and trade-offs instead of silently picking one. For a learning project, the dataset, base model and primary framework are choices; exact external IDs and access are verified before the plan. It writes the plan only after the high-impact choices are settled, then asks you to switch modes when you want it built. If it tries to change a file or run a command anyway, its turn stops there and the CLI says to press Shift+Tab to build; that stop is the harness's, so it holds for any model. `ask` asks before every edit and command. `bypassPermissions` suppresses routine prompts but does not weaken containment: opening a host application and protected code/config changes still require an explicit decision. (`acceptEdits`, the earlier name, means `default`.) Persistent `allow`/`deny` rules in config

**Outside content.** A prompt-injection attack can combine private data, content someone
else wrote, and a way to send data out. tinyharness tracks outside content: once a fetched
page, search results or a command's download is in the conversation, any call that could
send data to another machine is reviewed again. Auto mode gives the classifier only the
request and proposed call—not the
untrusted output—so it can allow aligned work or return a recoverable block; hard boundaries
and repeated uncertainty still go to the user. The review has two stages: a quick check on
the small model that can only allow, then a careful review that reasons first (your session's
model, or `[models.roles] safety_review`). It rates a call's risk apart from whether your
messages call for it, so low-risk work needs no explicit request; verdicts are cached per call
and request. Network tools lose the automatic public-page allowance, except for pages on a
host you named or on the sandbox allowlist, and leaving the sandbox says why it's asking.
Commands made entirely of known local readers such as `which`, `ls` and local Git inspection
stay quiet because they cannot send anything. So do plain downloads: `curl`/`wget` GETs of
literal URLs, read-only `gh` commands and text filters like `jq`, which send an allowed host
nothing but the URL. Request bodies, uploads, headers, scripts and new hosts are reviewed. In
auto mode, a command that needs one more host can be approved by the review for that command
only; with outside content present, you decide.
Sites you approved (config rules, "don't ask
again", an extension's approved needs) stay allowed, sandboxed commands stay contained
(they reach only the allowlist and can't read credential-shaped files), and `/clear` starts over.
`/sandbox` shows where the outside content came from.

## Sandbox

`Bash` runs with both filesystem and network isolation: native Seatbelt on macOS, or the optional [Anthropic Sandbox Runtime](https://github.com/anthropics/sandbox-runtime) on Linux, Windows and as a fallback. It writes only inside the project, session scratch directory, configured roots and a private tinyharness cache; cannot read credential files; reaches only explicitly trusted hosts; and cannot hand work to host applications. Common installers use private cache locations, including pip, uv, npm, pnpm, Yarn, Playwright, Puppeteer, Cypress, Hugging Face and Torch; temporary files stay in the session scratch directory. Those caches are separate from your own terminal's, so something downloaded or compiled in the sandbox is downloaded or compiled again outside it, and a virtualenv made in the sandbox may use a Python stored in the sandbox cache (Bash says so when it does). `[sandbox] share_caches = ["uv", "pip", "npm", "huggingface", "torch"]` opts any of them into your own cache locations instead; in exchange, a sandboxed command can change what those tools load later outside the sandbox. A tool that hard-codes a folder like `~/.cache/<tool>` is blocked there: the CLI shows a `sandbox blocked` row naming the folder, and the model is told to point the tool's own variable under `$XDG_CACHE_HOME` and remember it. Installing what the project already declares (`uv sync`, `uv pip install -e .`, `pip install -r requirements.txt`, `npm ci`, `bundle install`) runs without asking. Naming a new package, tool, bootstrap script, browser runtime, model, dataset or container is reviewed in auto mode and asks in other execution modes, including `bypassPermissions`; plan mode blocks it. In auto mode, an uncertain or denied classifier decision goes back to the agent so it can recover without interrupting the user; three consecutive blocks or twenty total blocks fall back to a person so the agent cannot loop forever. Classifier approval is identified in the UI and session log. Approval can cover that kind of install (Python packages, npm packages, ...) for the session; piped installers and PEP 723 scripts are approved per command, installs inside `sh -c`, `eval` and `exec` are found too, and direct use of an installed executable stays quiet. `[permissions] allow = ["Bash(capability:software:Python package)"]` pre-approves a kind. Installing the project's declared dependencies from PyPI or npm, in a command whose other steps only use non-executing setup (`cd`, plain environment variables, making a venv, activating a virtualenv inside the project) or read locally, doesn't count as outside content: it is authorized project setup inside the sandbox. Other sourced scripts and shell command substitutions are executable and keep their downloads marked as outside content. Installing a new package by name does too, because its build and its code can print untrusted text or change project files. Required network destinations are approved separately. A bare Python install reuses an active virtualenv or Conda environment only when that environment is writable inside the sandbox. Otherwise tinyharness creates and activates `.venv` in the project. Explicit pip or uv destinations such as `--target`, `--prefix`, `--root`, `--user` and `--system` are preserved and still constrained by the sandbox. Git hooks and config, `.tinyharness/`, editor tasks and settings (`.vscode/`, `.idea/`), `.envrc`, `.gitmodules`, `.mcp.json` and agent settings, commands and definitions stay read-only: each can run code outside the sandbox later. Unix sockets work only in folders the command may write (ZMQ `ipc://`, multiprocessing, language servers); sockets elsewhere, such as Docker's, stay closed. On macOS a single loopback port can't be closed again once loopback is allowed, so `allow_localhost` opens every local port. In auto mode the review decides loopback even after outside content was read, and remembers it for the session. Approving a host that belongs to a known multi-host service (Hugging Face: `huggingface.co`, its LFS CDNs and `*.xethub.hf.co` storage) opens that exact set. The proxy refuses an allowed name or literal address that is or resolves to a loopback, link-local (including cloud metadata) or unspecified address, and connects to the address it checked. Bash has no model-controlled escape switch. When a proxy-aware download is redirected to a new destination, the result names that exact host and the model may request it with `network_host`; the user can allow it once or for the session while the command remains sandboxed. An explicitly invoked skill can declare a finite exact destination set; its first required destination still asks, but the session option shows and grants that exact reviewed set, avoiding one failure and prompt per redirect or service. It never accepts wildcards. Stable destinations belong in `[sandbox] allow_hosts`; `[]` disables network access. HTTPS allowlisting controls destinations, not content, so an approved host may receive non-secret project data readable by the command. Bash uses ordinary shell semantics: use `&&` for dependent steps and `;` for independent probes. Unknown nonzero child exits are shown neutrally; known test/build failures, timeouts and sandbox blocks remain prominent. For CI and unattended work, `[sandbox] required = true` fails session startup when no backend is active. `/sandbox` shows the active policy and turns the sandbox on or off for the session (`/sandbox off`, `/sandbox on`; `[sandbox] enabled` sets the default).

**With the sandbox off**, commands run as you: they can read and write any folder and reach any host, and installs use your normal caches. Commands that only read run without asking in `default` and `auto`. Other commands ask in `default` and go to auto mode's review in `auto`, which decides as it does for any other call. What the sandbox used to stop asks you, in every mode except `bypassPermissions`: deleting files, `sudo`, anything touching credential files (`~/.ssh`, `.env`, cloud and `gh` configs), and changing files that run code later (git hooks, editor settings, `.envrc`, agent settings). A "don't ask again" or `allow = ["Bash"]` rule doesn't cover those. Outside content still matters: a command that looks like a download marks it, since no proxy records which hosts were reached, and afterwards calls that could send data out are reviewed. Having no backend at all behaves the same, with a warning at startup. Switching changes the Bash tool's description, so the next request rebuilds the prompt cache once. `Open` presents an artifact inside tinyharness without leaving containment; its separate external mode launches a host application and always asks.

Exact hosts approved for the session accumulate in later command policies. This lets a client
cross several service or redirect hosts without losing earlier grants, while one-command approvals
remain one-command and model-supplied wildcards remain invalid.
An allowlisted host is still an outbound channel: after outside content enters the conversation,
network-capable Bash calls ask before running even when their destination is already allowed.

Tests, previews and development servers can request `network_host="localhost"`. The scoped
approval enables loopback while the command remains sandboxed; it does not open external network
access. Credential-shaped pytest fixtures are readable only beneath the session's private temporary
directory, while real credential locations remain blocked.

## Security

API keys never reach the model: tools refuse to read secret-shaped files (`.env`, SSH keys, …), configured key values are masked in every tool result, and a key pasted into the chat is caught before sending. Hooks get no keys in their environment, and keys are masked out of the input they're sent; session logs are private to you and masked the same way; symlinks are resolved before permission checks

## Context management

Three tiers: cap oversized messages → stub old tool output → summarize history. Old tool output is cleared only when it pays: once the cache has expired anyway, or near the window in batches of at least 20k tokens. The prompt cache isn't rebuilt every turn

## Skills

`SKILL.md` directories, loaded progressively and portable across tools that use this format. A
skill that depends on optional harness capabilities can declare `requires-extensions: [agents]`;
explicit invocation activates those extensions for the current session before the model starts it.
A skill that uses a known finite set of data services can declare exact hosts with
`requires-network: [api.example.com, assets.example.com]`; tinyharness offers that reviewed set
as one session approval while keeping one-command approval limited to the requested host

## Subagents and teams

Isolated context windows; built-in `explore` / `plan` / `general-purpose` / `researcher`, plus your own in Markdown, each on any model (`model: local:planner`) with its own step limit. `Team` runs a dependency graph of tasks whose size the model chooses (3-5 teammates suit most goals). Up to `max_parallel_agents` subagents (default 20, or `TINYHARNESS_MAX_PARALLEL_AGENTS`) run at once across the session, Agent calls and teammates together; past that an Agent call fails so the model can wait, and Team tasks wait their turn: independent tasks run in parallel, each teammate gets the results of the tasks it depends on, the user's request and its working directory, and teammates can message each other. Writing calls take turns on a shared lock; reads (including read-only shell commands) don't wait

## Hooks and plugins

Shell or Python hooks on 5 lifecycle events; plugins bundle skills, agents and hooks

## Providers

Each provider through its own current API: Anthropic Messages, OpenAI Responses, Gemini Interactions, and Chat Completions for the rest, plus any gateway. Reasoning carries across tool calls (signed/encrypted state is sent back unchanged). No per-model special cases: models, context windows and capabilities come from each provider's own model list

## Terminal UI

The input box stays up while the agent works: type to steer it (your message arrives before its next step), Ctrl+C to stop it. Built-in commands work at any time: settings like `/effort` and `/mode` apply from the agent's next step (a new `/model` from its next turn), and `/clear`, `/sessions` and `/compact`, which replace the conversation the agent is adding to, run as soon as it finishes. `/btw <question>` asks a side question at any time, even mid-turn: it's answered from the conversation so far, without tools, in one call that is mostly read from the cache, and neither question nor answer is added to the conversation. The answer shows above the input box until Esc (or your next message) closes it, leaving nothing in the scrollback; `/output` prints it in full. When a turn ends, the empty input box suggests your likely next message (Tab takes it to edit or send); `prompt_suggestions = false` turns that off. The model's reasoning is one line (`∴ Thought for 8s`) until `ctrl+t` or `/thinking` shows it in full. One row per tool call, colored by what it does (looking, changing, running, the web, agents), with its result on the right: `+14 −3`, `212 passed`, `exit 1` and the error under it. Subagents hang under their call. Each turn ends with a summary (time, files changed, whether the tests passed, tokens, cost), and a status bar shows the model, sandbox, git branch, context and cost. `ctrl+o` expands the last tool's output (or what running subagents are doing) above the input box and collapses it again, `/output` prints it in full; file paths are clickable; themes: `system` (the default: asks your terminal whether its background is light or dark), `light` or `dark`, set with `/theme` or `theme =` in the config; `NO_COLOR` is honored

## Two interfaces

Interactive CLI (streamed Markdown, live tool status, `/` autocomplete, pickers, mode-colored input box) and an async Python SDK, both consuming the same typed event stream

## Resume

Every conversation is saved losslessly as append-only JSONL in `~/.tinyharness/projects/<project>/`, never inside your repo (so it can't be committed) and where sandboxed commands can't write. Trace schema 1 adds session, turn, agent, model-call, routing-decision and tool-call correlation plus the harness version. Main-agent and subagent loop calls record request, first-token and completion timing; permission, tool, compaction and total-turn timings are also recorded in milliseconds. Classifier, compaction and web-digest calls do not yet guarantee complete model-call or token/cost records. A `routing_decision` records the configured/static model choice in this release, not automatic routing. `turn_complete` means the agent loop finished, `evaluation` is a bounded post-turn observer result, and `turn_end` means session-level post-processing finished. Evaluators run before the final `Done` reaches an SDK consumer but cannot change the completed trajectory. `-c` continues the latest conversation. `-r` opens a picker showing titles, recency, message counts and models; `-r words` searches those titles, so session ids remain an automation detail. Inside the CLI, `/sessions` searches and switches conversations without a restart. An OS-held lock prevents two tinyharness processes from appending to the same session log; separate conversations can remain open in separate terminals. On resume the harness checks what changed while you were away (files the agent read, git HEAD or branch, AGENTS.md): stale file contents in the history are replaced by a stub, and the model is told what changed before your next message
