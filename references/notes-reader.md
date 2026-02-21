# Notes Reader Contract

Read only `.branch-context/<branch_key>/notes.md`.

## Goal

Extract the canonical notes signal for load-time synthesis.

## Output Format

Return only JSON with this schema:

```json
{
  "objective": "string",
  "decisions": ["string"],
  "constraints": ["string"],
  "risks": ["string"],
  "workingSetHints": ["string"]
}
```

## Extraction Rules

- `objective`: from `## Active Objective (generated)`.
- `decisions`: bullet items from `## Key Decisions (generated)`.
- `constraints`: only explicit constraints from notes text; do not infer.
- `risks`: from explicit risk/open-question statements if present.
- `workingSetHints`: important file/workstream hints from generated sections.

If a field is missing, return an empty value (`""` or `[]`).
