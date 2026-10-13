# Configuration

tinyharness needs an API key for at least one provider. The first run asks for one
(with a link to where to get it), checks it with the provider and keeps it in
`~/.tinyharness/auth.json`, one entry per provider, readable only by you and never by
the agent's tools. Switching to a provider with no key (`/provider`, `/model`) asks
for its key the same way, and `/key` adds or replaces a saved key. After that, everything is
optional: `/model` switches models (and remembers your choice), and one file,
`~/.tinyharness/config.toml`, holds any defaults you want.

One key is resolved per provider. The precedence is an SDK `SessionOptions(api_key=...)`
override, the process environment, a development `.env`, then `auth.json`. `.env` is read
only from a tinyharness source checkout; installed CLI and SDK users do not pick up project
dotenv files. Saved and dotenv keys are not copied into `os.environ`. `/whoami` shows the
selected source without exposing the key.

Everything tinyharness keeps for you is in one folder, `~/.tinyharness/`:
`config.toml`, `auth.json`, your `skills/`, `agents/`, `plugins/` and `hooks.toml`, and
its own state and session logs. (Earlier versions used `~/.config/tinyharness/`; its
files are moved on the first run.) See
[`examples/config.toml`](https://github.com/anujdutt9/tinyharness/blob/main/examples/config.toml). API keys never go in the config file.

**Providers.** Each one is used through its own current API:

| API format | Providers |
|---|---|
| Anthropic Messages | Anthropic |
| OpenAI Responses | OpenAI, xAI |
| Gemini Interactions | Google Gemini |
| OpenAI Chat Completions | OpenRouter, DeepSeek, Mistral, Groq, and custom gateways |

[`llm/providers.toml`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/llm/providers.toml) holds only how to connect to
each one. Models are listed live from the provider (cached for a day), and whatever
the provider reports about a model (context window, output limit, reasoning support)
is used as-is; nothing is special-cased by model name. When a provider doesn't say
(OpenAI doesn't publish context windows), a default applies, and if a request is
ever rejected as too long, the history is summarized and the request retried.

A gateway is one table in the config file:

```toml
model = "gateway:claude-sonnet-5"

[providers.gateway]
base_url = "https://llm.internal.example.com/v1"
api_key_env = "GATEWAY_API_KEY"
api = "chat"   # or "responses", "anthropic", "gemini"
```

**Self-hosted models (vLLM, Ollama, llama.cpp, SGLang).** Any OpenAI-compatible
server is a provider; leave out `api_key_env` when it needs no key. tinyharness reads the
served models and their context windows from `/v1/models` (vLLM's `max_model_len`), so
compaction triggers at the server's real limit: once the history leaves less room than the
model's output limit (`max_output` in a `models` entry, else what the server lists, else
32,000 tokens; at most half the window). Two more settings on a `models` entry help with
self-hosted models: `tokenizer = "<Hugging Face repo>"` counts tokens with the model's own
tokenizer (needs the `tokenizers` package: `tinyharness[tokens]`), so what compaction keeps
fits its budget; without it, counts are estimated and calibrated against the server's own
counts as the session goes. `images = false` says the model takes no image input (vLLM's
`--language-model-only`, a text-only model): images are sent as a note saying so instead. Left
unset, the first refusal of an image is enough; the model gets notes from then on. Agents can
run on different servers:

```toml
model = "local:planner"

[providers.local]
label = "vLLM on my GPU box"
base_url = "http://127.0.0.1:8000/v1"
# api_key_env = "VLLM_API_KEY"         # only if the server was started with --api-key
extra_body = { temperature = 0.2 }      # merged into every request
```

For tool use, start vLLM with `--enable-auto-tool-choice --tool-call-parser <parser>`
(e.g. `hermes` for Qwen2.5/Qwen3 instruct models, `qwen3_coder` for Qwen3-Coder). A server
without a tool-call parser works too: the first request to a model finds out how it calls
tools, and a model that writes its calls as text gets the tools in its prompt instead
(`tool_calls = "text"` forces it). An agent file can then pick a model with
`model: local:worker`.

Background model work is configured by role rather than one catch-all model:

```toml
[models.roles]
safety_classifier = "claude-haiku-4-5"
search_digest = "claude-haiku-4-5"
compaction = "claude-sonnet-5"
```

Role values are model IDs on the active provider. Do not include a provider prefix such
as `openai:`; cross-provider background roles are not supported in this release.
Unset classifier and search roles fall back to the active provider's `small_model` for
compatibility. An unset compaction role uses `compaction_model`, then the active model.
The same resolved compaction role is used by the lead and subagents.

| Location | Contents |
|---|---|
| `~/.tinyharness/` | your skills, agents, plugins, `hooks.toml`, `AGENTS.md`; `projects/<project>/` holds each project's conversation logs and scratch files |
| `<project>/.tinyharness/` | project skills, agents, plugins and hooks: only what you put there, to commit and share. tinyharness never writes into your project |
| `<project>/AGENTS.md` | project instructions added to the system prompt |
