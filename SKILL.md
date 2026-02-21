---
name: branch-context-manager
description: Persist and restore branch-scoped working context with `update`, `load`, and `delete` commands backed by `.branch-context/<branch_key>/`. Use when switching machines/sessions, resuming paused PR work, or cleaning up branch context after merge.
---

# Branch Context Manager

Use `scripts/branch_context_manager.sh` to manage branch context state.

## Commands

```bash
# Capture context from git + transcript and commit it
bash scripts/branch_context_manager.sh update

# Capture context without committing
bash scripts/branch_context_manager.sh update --no-commit

# Force runtime adapter (auto|cursor|claude|codex|opencode|none)
bash scripts/branch_context_manager.sh update --runtime codex

# Read context summary for the current branch
bash scripts/branch_context_manager.sh load

# Read context as JSON payload
bash scripts/branch_context_manager.sh load --json

# Delete branch context and commit removal
bash scripts/branch_context_manager.sh delete
```

## Runtime and Transcript Selection

- Explicit transcript path always wins: `--transcript-path /path/to/file.jsonl`
- Runtime override: `--runtime <value>`
- Environment fallback: `BRANCH_CONTEXT_RUNTIME=<value>`
- `none` disables transcript discovery and records only git-derived context

## Multi-Agent Load Pattern

For `load` synthesis workflows, apply these contracts:

- `references/notes-reader.md`
- `references/plans-reader.md`
- `references/todo-reader.md`
- `references/coordinator-schema.md`

## Safety

- Never store secrets, credentials, or tokens in context files.
- Treat `.branch-context/` as repo-visible state and review before commit.
- Preserve manual sections only between `<!-- manual-... -->` markers.
