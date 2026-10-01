# Skills

A skill is a reusable set of instructions, saved as a directory with a `SKILL.md`,
that you run with `/<name>`. When it runs, its body becomes the prompt for that turn.

```
release-notes/
  SKILL.md            required: frontmatter + instructions
  references/         optional: docs the model reads on demand
    style-guide.md
  scripts/            optional: helpers the skill refers to
```

```markdown
---
name: release-notes
description: Write release notes for recent changes. Args: a git range or topic.
allowed-tools: [Read, Grep, Glob, Write]
---

You are a technical writer.

1. If arguments were given, treat them as the scope. Otherwise use CHANGELOG.md.
2. Read `references/style-guide.md`.
3. Group changes into Features / Fixes / Internal.
4. Write `release-notes-<YYYY-MM-DD>.md`.
```

| Frontmatter | Default | Meaning |
|---|---|---|
| `name` | directory name | the `/<name>` command |
| `description` | — | shown by `/skills`, and to the model (its first ~240 characters) so it knows when to use the skill |
| `allowed-tools` | all | tools the skill expects; informational for now |
| `model` | session model | reserved for per-skill model overrides |

## Where skills live

| Location | Scope |
|---|---|
| `~/.tinyharness/skills/<name>/` | every project |
| `<project>/.tinyharness/skills/<name>/` | this project only; commit it to share with your team |
| `<plugin>/skills/<name>/` | shipped inside a [plugin](extending.md#plugins) |

If two skills have the same name, the one in your home directory wins. That way a
cloned repo can't silently replace a skill you trust. Set `TINYHARNESS_HOME` to move
the home directory.

## Progressive disclosure

Skills cost nothing until you use them:

1. **At startup** tinyharness reads only each skill's name and description.
2. **When you run `/<name>`** the body is sent to the model.
3. **Bundled files** are read one at a time, when the step that needs them comes up.
   The skill's directory is readable without permission prompts.

The system prompt lists each skill's name, the start of its description and where its
SKILL.md is. When a task matches (a design skill for a web page, say), the model loads the
skill by reading that file with Read; there is no separate tool. Subagents and teammates
see the same list. You can also point it at one ("use the claude-design-skill") or run it
yourself with `/<name>`. The list is about 80 tokens per skill, cached with the rest of
the prompt. `skill_tool = false` in the config (or `--no-skill-tool`) leaves skills to
`/<name>` only.

## Running skills

```text
» /skills
» /release-notes v1.2..v1.3
```

```python
from tinyharness import SessionOptions, invoke_skill

async for event in invoke_skill("release-notes", "v1.2..v1.3", SessionOptions()):
    ...
```

## Writing good skills

- **Numbered steps.** Models follow numbered lists more reliably than prose.
- **Say where output goes.** For example, "write `report.md` in the working directory".
- **Say what arguments mean.** For example, "if arguments were given, treat them as ...".
- **Keep reference files lazy.** Each file you read costs context. Tell the model which step needs which file.
- **Delegate big stages.** For long pipelines, the `Agent` tool runs a stage in a fresh context and returns only the result.

## Importing skills from other tools

Skills written for Claude Code or another tool that uses `SKILL.md` work unchanged:

```bash
tinyharness --import-skills              # from ~/.claude/skills
tinyharness --import-skills ./my-skills  # from any directory
tinyharness --import-skills --symlink    # link instead of copy
```

A skill that relies on a tool tinyharness doesn't have (for example `Bash` or `WebFetch`) will fail at that step.
