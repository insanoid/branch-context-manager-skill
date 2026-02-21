# Plans Reader Contract

Read only `.branch-context/<branch_key>/plans/*.md`.

## Goal

Extract latest plan state, milestones, and unresolved items.

## Output Format

Return only JSON with this schema:

```json
{
  "planSummaries": [
    {
      "slug": "string",
      "title": "string",
      "keyMilestones": ["string"],
      "openItems": ["string"]
    }
  ],
  "globalMilestones": ["string"],
  "globalOpenItems": ["string"]
}
```

## Extraction Rules

- Treat each file as the latest version for its plan identity.
- Prefer explicit checklist/todo/open-question lines.
- Do not invent milestones or completion states.
- Keep each summary concise and factual.
