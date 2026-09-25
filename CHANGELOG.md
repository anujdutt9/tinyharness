# Changelog

All notable changes to tinyharness. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/): while on 0.x, minor versions may break
the SDK, and every break is listed here.

## 0.1.0 (unreleased)

First release.

### CLI
- Interactive coding agent with streamed Markdown, live tool status, `/` command
  autocomplete, pickers for model, provider, effort and permission mode.
- Permission prompts with diff previews; "don't ask again" per project or directory;
  Shift+Tab cycles `default` → `acceptEdits` → `plan`; persistent allow/deny rules.
- Guide on first run when no API key is set; keys live in `~/.config/tinyharness/.env`.

### Providers
- Native APIs: Anthropic Messages, OpenAI Responses (OpenAI, xAI), Gemini
  Interactions, and Chat Completions (OpenRouter, DeepSeek, Mistral, Groq, gateways).
- Reasoning carries across tool calls; one `effort` setting for every model.
- Models, context windows and capabilities come from each provider's model list.
- Models are checked before switching; context overflows are compacted and retried.

### SDK
- `Session`, `query` and a typed event stream shared with the CLI.
- Custom tools, hooks, skills, subagents and plugins.

### Security
- API keys never reach the model: secret files are unreadable by tools, configured
  key values are masked in tool results, and keys pasted into the chat are caught.
