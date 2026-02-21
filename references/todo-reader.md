# Todo Reader Contract

Read only `.branch-context/<branch_key>/todo.md`.

## Goal

Extract operational status from generated todo sections.

## Output Format

Return only JSON with this schema:

```json
{
  "active": ["string"],
  "blocked": ["string"],
  "done": ["string"],
  "criticalBlockers": ["string"]
}
```

## Extraction Rules

- Parse only generated sections unless user explicitly asks to include manual section.
- `criticalBlockers` should include only blocker items that clearly block progress.
- Keep output deterministic; no guessed priorities.
