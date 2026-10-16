# Configuration

tinyharness needs a connection to at least one provider. The first run asks where your
models come from, then offers the methods that provider supports. OpenRouter can open a
PKCE-protected browser sign-in (or display a one-time-code flow for a remote machine);
GitHub Copilot uses GitHub's device sign-in and the models included in that subscription.
ChatGPT plan uses Continue with ChatGPT for eligible Plus and Pro accounts, with no OpenAI
API key or API billing account required. OpenAI API is a separate provider choice. Other
hosted providers can use an API key, and a local server may need no credential.
Connections are checked before use and saved in `~/.tinyharness/auth.json`, readable only
by you and never by the agent's tools. `/connect` adds or replaces a connection, while
`/key` is the direct API-key path. `/provider` changes where models come from, then
`/model` lists only that provider's models and remembers your choice;
`~/.tinyharness/config.toml` holds any defaults you want.

The hosted choices include Anthropic (Claude), OpenAI API (GPT), ChatGPT plan, Google
(Gemini), OpenRouter, GitHub Copilot, xAI, DeepSeek, Mistral and Groq. A direct provider
connection uses that provider's API account unless setup explicitly offers subscription sign-in.

ChatGPT sign-in opens OpenAI in the system browser and returns to a loopback callback on
`127.0.0.1`. tinyharness validates the signed OpenID identity and granted plan permission,
then stores separate renewable profiles in `auth.json`; access, refresh and identity tokens
never enter the model context. `/connect chatgpt` can reconnect a saved account or add a
different account/workspace. Its model picker comes from that account's live plan catalog.

LiteLLM appears as a first-class gateway choice for company cost tracking, routing and
policy enforcement. Setup asks for the proxy's OpenAI-compatible `/v1` URL and optional
key, then loads its live model list and saves the connection. It works with local and
company-hosted LiteLLM Proxy deployments.

No configuration file is needed for these connections. On first run—and again from
`/connect`—local runtimes and gateways offer the standard or saved endpoint alongside an
“enter another endpoint” option. Reconfiguration can keep, replace or remove a saved
server key independently. The TOML examples below remain useful for reproducible team
defaults and advanced overrides.

Ollama, LM Studio, llama.cpp, vLLM and SGLang appear in setup with their standard localhost
ports. tinyharness asks the server for its live model list, lets you choose a model, and
checks it with a small request. When the standard endpoint cannot be reached, setup asks
for a remote/custom URL and optional server key and remembers them in the protected
`auth.json`. You can also set `base_url` in `[providers.<name>]`; explicit configuration
wins over the remembered endpoint.

For Ollama, discovery also reads `/api/show` for every installed model. Models that do not
accept native API tools automatically use tinyharness's text-tool compatibility mode, and
vision support is recorded from the same capability list.

One credential is resolved per provider. For API keys, precedence is an SDK `SessionOptions(api_key=...)`
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
| OpenAI Responses | OpenAI API, ChatGPT plan, xAI |
| Gemini Interactions | Google Gemini |
| OpenAI Chat Completions | OpenRouter, DeepSeek, Mistral, Groq, and custom gateways |

GitHub Copilot is mixed-format: its live catalog declares whether each available model
uses Chat Completions, Responses or Anthropic Messages. LiteLLM uses its OpenAI-compatible
Chat Completions interface.

The Copilot connection follows the compatibility flow used by Pi and OpenCode: GitHub
device authorization, a short-lived Copilot service token, and GitHub's live model
catalog. Auto, the only choice on Free and Student plans, is not a model id: tinyharness
asks GitHub which model Auto picks for the session (as VS Code's Copilot Chat does) and
sends that model. It keeps tinyharness's own agent loop, but these service endpoints are not a
documented standalone public model API and may require maintenance when GitHub changes
its Copilot clients.

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

`/effort` reads the active model's catalog metadata before showing its choices. When the
provider advertises exact effort levels, only those levels can be selected. A model that
advertises no reasoning control gets only `default` and `none`. Sparse self-hosted catalogs
are treated as unknown rather than unsupported: the other levels remain visible with a
warning and are sent only when the user deliberately selects one. Known-unsupported effort
parameters are omitted from requests.

| Location | Contents |
|---|---|
| `~/.tinyharness/` | your skills, agents, plugins, `hooks.toml`, `AGENTS.md`; `projects/<project>/` holds each project's conversation logs and scratch files |
| `<project>/.tinyharness/` | project skills, agents, plugins and hooks: only what you put there, to commit and share. tinyharness never writes into your project |
| `<project>/AGENTS.md` | project instructions added to the system prompt |
