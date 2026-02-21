from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import branch_context as bc


def test_branch_key_normalizes_separators() -> None:
    assert bc.branch_key("feat/cache rebuild") == "feat__cache__rebuild"
    assert bc.branch_key("___") == "unknown-branch"


def test_clean_text_removes_noise() -> None:
    raw = "L10: <tag>Hello</tag>\n\n<code_selection>ignore me</code_selection>   world"
    assert bc.clean_text(raw).replace(" ", "") == "Hello\n\nworld"


def test_parse_frontmatter_todos() -> None:
    frontmatter = """name: test-plan
todos:
  - id: one
    content: First task
    status: pending
  - id: two
    content: Blocked task
    status: blocked
"""

    todos = bc.parse_frontmatter_todos(frontmatter)

    assert [todo.item_id for todo in todos] == ["one", "two"]
    assert [todo.status for todo in todos] == ["pending", "blocked"]


def test_classify_todos_groups_statuses() -> None:
    record = bc.PlanRecord(
        slug="sample",
        source_path=Path("/tmp/sample.plan.md"),
        title="Sample",
        updated_at=0.0,
        content="# Sample",
        todos=[
            bc.PlanTodo(item_id="a", content="Do A", status="pending"),
            bc.PlanTodo(item_id="b", content="Do B", status="blocked"),
            bc.PlanTodo(item_id="c", content="Do C", status="done"),
        ],
    )

    active, blocked, done = bc.classify_todos({"sample": record})

    assert active == ["[sample] Do A"]
    assert blocked == ["[sample] Do B"]
    assert done == ["[sample] Do C"]


def test_load_context_payload_reads_current_layout(tmp_path: Path) -> None:
    branch = "feat/demo"
    key = bc.branch_key(branch)
    context_dir = tmp_path / ".branch-context" / key
    plans_dir = context_dir / "plans"
    plans_dir.mkdir(parents=True)

    (context_dir / "notes.md").write_text(
        """## Active Objective (generated)
Ship the feature

## Key Decisions (generated)
- Use typed Python tooling
"""
    )
    (context_dir / "todo.md").write_text(
        """## Active (generated)
- [ ] [sample] Do A

## Blocked (generated)
- [ ] [sample] Do B

## Done (generated)
- [x] [sample] Do C
"""
    )
    (plans_dir / "sample.md").write_text("# Sample Plan\n")

    payload = bc.load_context_payload(repo_root=tmp_path, branch=branch)

    summary = payload["summary"]
    assert isinstance(summary, dict)
    assert summary["objective"] == "Ship the feature"
    assert summary["plan_count"] == 1
