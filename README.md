# Branch Context Manager Skill

Branch Context Manager is a standalone agent skill that saves and restores branch-specific context across sessions.

It writes to:

- `.branch-context/<branch_key>/notes.md`
- `.branch-context/<branch_key>/todo.md`
- `.branch-context/<branch_key>/plans/*.md`

## Install This Skill

Install `branch-context-manager` from this repository:

```bash
# Install in the current project
npx skills add insanoid/branch-context-manager-skill --skill branch-context-manager

# Install globally for specific agents
npx skills add insanoid/branch-context-manager-skill --skill branch-context-manager -g -a codex -a claude-code

# Verify the skill is available
npx skills list | rg branch-context-manager
```

## Repository Layout

```text
.
├── SKILL.md
├── README.md
├── scripts/
│   ├── branch_context_manager.sh
│   └── branch_context.py
└── references/
    ├── notes-reader.md
    ├── plans-reader.md
    ├── todo-reader.md
    └── coordinator-schema.md
```

## Concrete Example

Scenario: You pause work on `feat/cache-rebuild` on laptop A and continue on laptop B.

1. On laptop A, capture branch context:

```bash
bash scripts/branch_context_manager.sh update
```

2. Push your branch (context files are committed with your branch work).

3. On laptop B, pull the branch and load context:

```bash
bash scripts/branch_context_manager.sh load
```

4. You now get objective, decisions, active blockers, and plan references from `.branch-context/feat__cache-rebuild/`.

## Execution Flow (ASCII)

```text
             update
+-------------------------------------------+
| branch_context_manager.sh                 |
|   -> branch_context.py update             |
+------------------------+------------------+
                         |
                         v
          +--------------+-------------------+
          | Runtime + transcript discovery   |
          | (cursor/claude/codex/opencode)  |
          +--------------+-------------------+
                         |
                         v
          +--------------+-------------------+
          | Build context artifacts          |
          | notes.md / todo.md / plans/*.md  |
          +--------------+-------------------+
                         |
                         v
          +--------------+-------------------+
          | Optional commit                 |
          | chore(context): update...       |
          +---------------------------------+

             load / delete
+-------------------------------------------+
| load   -> reads .branch-context/<branch_key>/ |
| delete -> removes .branch-context/<branch_key>/
+-------------------------------------------+
```

## Commands

```bash
# Update context (auto transcript discovery + commit)
bash scripts/branch_context_manager.sh update

# Update without commit
bash scripts/branch_context_manager.sh update --no-commit

# Override runtime adapter
bash scripts/branch_context_manager.sh update --runtime codex

# Provide explicit transcript file
bash scripts/branch_context_manager.sh update --transcript-path ~/.codex/sessions/session.jsonl

# Load as markdown summary
bash scripts/branch_context_manager.sh load

# Load as JSON
bash scripts/branch_context_manager.sh load --json

# Delete context
bash scripts/branch_context_manager.sh delete
```

## Development

### Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

### Setup

```bash
uv sync --group dev
```

### Quality Checks

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy scripts tests

# Tests
uv run pytest
```

### One Command

```bash
make check
```

## Continuous Integration

GitHub Actions validates this repository on pull requests and pushes to `main`:

- `ruff` linting
- `mypy` type checks
- `pytest` test suite
- CLI smoke check (`branch_context_manager.sh --help`)
- skills install/discovery smoke check (`npx skills add . --list`)

## Release Management

Releases are automated with `semantic-release` and run on pushes to `main`.

- evaluates commit history using Conventional Commits
- computes the next semver version
- updates `CHANGELOG.md`
- creates a GitHub Release with generated notes
- creates a release commit (`chore(release): <version> [skip ci]`)

### Commit Format

Use Conventional Commit messages (examples):

- `feat: add opencode runtime discovery fallback`
- `fix: avoid absolute path leakage in notes`
- `docs: clarify branch_key mapping`
- `chore(deps): update semantic-release plugins`

### Local Dry Run

```bash
npm ci
npm run release:dry-run
```

## Notes

- Legacy `.context/<branch_key>/` data is auto-migrated to `.branch-context/<branch_key>/`.
- Manual sections in `notes.md` and `todo.md` are preserved between updates.
- Avoid storing credentials or sensitive personal data in context snapshots.
