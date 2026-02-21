# Coordinator Schema

Merge outputs from the notes, plans, and todo readers into one canonical object.

## Canonical Object

Return only JSON:

```json
{
  "branch": "string",
  "branchKey": "string",
  "branchContextEntityName": "BranchContext::<repo>::<branch_key>",
  "objective": "string",
  "decisions": ["string"],
  "constraints": ["string"],
  "risks": ["string"],
  "activeTodos": ["string"],
  "blockedTodos": ["string"],
  "doneTodos": ["string"],
  "planMilestones": ["string"],
  "planOpenItems": ["string"],
  "generatedAtUtc": "ISO-8601 timestamp"
}
```

## Merge Rules

- Preserve explicit facts from reader outputs.
- Keep `branchKey` consistent with the normalized context directory key.
- Deduplicate exact duplicates; preserve semantic differences.
- Prefer shorter, concrete statements over long prose.
- If a field is unavailable, emit empty string/array.

## Memory Upsert Mapping

Transform canonical object into memory observations:

- `Objective: <objective>`
- `Decisions: <d1>; <d2>; ...`
- `Constraints: <...>`
- `Risks: <...>`
- `ActiveTodos: <...>`
- `BlockedTodos: <...>`
- `PlanMilestones: <...>`
- `PlanOpenItems: <...>`
- `SnapshotAt: <generatedAtUtc>`

## Failure Policy

- If any reader result is missing or invalid: fail with explicit error.
- If memory upsert fails: fail-fast; do not silently continue.
