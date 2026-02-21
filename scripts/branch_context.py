#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MAX_TRANSCRIPT_MESSAGES = 20
MAX_MESSAGE_CHARS = 260
DEFAULT_MANUAL_NOTE = "- Add branch-specific notes here."
DEFAULT_MANUAL_TODO = "- [ ] Add manual todo item here."
CONTEXT_ROOT_DIR = ".branch-context"
LEGACY_CONTEXT_ROOT_DIR = ".context"
MANUAL_NOTES_START = "<!-- manual-notes:start -->"
MANUAL_NOTES_END = "<!-- manual-notes:end -->"
MANUAL_TODO_START = "<!-- manual-todo:start -->"
MANUAL_TODO_END = "<!-- manual-todo:end -->"
SUPPORTED_RUNTIME_OVERRIDES = ("auto", "cursor", "claude", "codex", "opencode", "none")
KNOWN_CHAT_ROLES = {"user", "assistant", "system", "tool", "developer"}


class BranchContextError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    text: str
    raw_text: str


@dataclass(frozen=True)
class PlanTodo:
    item_id: str
    content: str
    status: str


@dataclass(frozen=True)
class PlanRecord:
    slug: str
    source_path: Path
    title: str
    updated_at: float
    content: str
    todos: list[PlanTodo]


@dataclass(frozen=True)
class RuntimeDetection:
    provider: str
    confidence: float
    reasons: list[str]


@dataclass(frozen=True)
class TranscriptCandidate:
    path: Path
    score: int
    modified_at: float


def run_command(command: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        detail = stderr or stdout or "command failed"
        raise BranchContextError(f"{' '.join(command)}: {detail}")
    return proc


def git_repo_root(explicit_repo_root: str | None) -> Path:
    if explicit_repo_root:
        root = Path(explicit_repo_root).expanduser().resolve()
        if not root.exists():
            raise BranchContextError(f"repo root does not exist: {root}")
        return root

    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], text=True, capture_output=True
    )
    if proc.returncode != 0:
        raise BranchContextError("not inside a git repository")
    return Path(proc.stdout.strip()).resolve()


def current_branch(repo_root: Path, override_branch: str | None) -> str:
    if override_branch:
        return override_branch.strip()

    proc = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    branch = proc.stdout.strip()
    if not branch or branch == "HEAD":
        raise BranchContextError("detached HEAD: pass --branch explicitly")
    return branch


def branch_key(branch: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "__", branch.strip())
    safe = safe.strip("_")
    return safe or "unknown-branch"


def clean_text(value: str) -> str:
    text = value.replace("\r\n", "\n")
    text = re.sub(r"<code_selection[^>]*>.*?</code_selection>", " ", text, flags=re.DOTALL)
    text = re.sub(r"^L\d+:\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"</?[a-zA-Z0-9_:-]+>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    return f"{value[: max_len - 1].rstrip()}…"


def display_path(value: Path, repo_root: Path) -> str:
    resolved = value.expanduser().resolve()
    try:
        return str(resolved.relative_to(repo_root).as_posix())
    except ValueError:
        pass

    home = Path.home().resolve()
    try:
        rel_home = resolved.relative_to(home).as_posix()
        return f"~/{rel_home}"
    except ValueError:
        return resolved.name


def display_transcript_source(transcript_path: Path | None, repo_root: Path) -> str:
    if not transcript_path:
        return "not found"
    return display_path(transcript_path, repo_root)


def cursor_project_slug(repo_root: Path) -> str:
    return repo_root.as_posix().lstrip("/").replace("/", "-")


def project_identifier_tokens(repo_root: Path) -> list[str]:
    raw_parts = [
        repo_root.name,
        repo_root.parent.name,
        repo_root.parent.parent.name if repo_root.parent.parent else "",
        repo_root.parent.parent.parent.name if repo_root.parent.parent.parent else "",
    ]
    tokens: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        for token in re.split(r"[^A-Za-z0-9]+", part.lower()):
            if len(token) < 3:
                continue
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
    return tokens


def normalize_runtime_value(raw_value: str, source: str) -> str:
    value = raw_value.strip().lower()
    if value not in SUPPORTED_RUNTIME_OVERRIDES:
        allowed = ", ".join(SUPPORTED_RUNTIME_OVERRIDES)
        raise BranchContextError(f"invalid runtime in {source}: {raw_value!r} (expected one of {allowed})")
    return value


def resolve_runtime_override(cli_runtime: str | None) -> tuple[str | None, str | None]:
    if cli_runtime is not None:
        normalized = normalize_runtime_value(cli_runtime, "--runtime")
        if normalized != "auto":
            return normalized, "cli"
        return None, None

    env_runtime = os.environ.get("BRANCH_CONTEXT_RUNTIME")
    if env_runtime:
        normalized = normalize_runtime_value(env_runtime, "BRANCH_CONTEXT_RUNTIME")
        if normalized != "auto":
            return normalized, "env"
    return None, None


def signal_confidence(base: float, signal_count: int) -> float:
    if signal_count <= 0:
        return 0.0
    confidence = base + ((signal_count - 1) * 0.08)
    return round(min(confidence, 0.99), 2)


def detect_runtime(repo_root: Path, cli_runtime: str | None) -> RuntimeDetection:
    override, source = resolve_runtime_override(cli_runtime)
    if override:
        return RuntimeDetection(
            provider=override,
            confidence=1.0,
            reasons=[f"explicit_{source}_override:{override}"],
        )

    cursor_reasons: list[str] = []
    if os.environ.get("CURSOR_AGENT"):
        cursor_reasons.append("CURSOR_AGENT")
    if os.environ.get("CURSOR_CLI"):
        cursor_reasons.append("CURSOR_CLI")
    if os.environ.get("TERM_PROGRAM", "").lower() == "cursor":
        cursor_reasons.append("TERM_PROGRAM=cursor")
    if cursor_reasons:
        return RuntimeDetection(
            provider="cursor",
            confidence=signal_confidence(0.78, len(cursor_reasons)),
            reasons=cursor_reasons,
        )

    claude_reasons: list[str] = []
    if os.environ.get("CLAUDE_PROJECT_DIR"):
        claude_reasons.append("CLAUDE_PROJECT_DIR")
    if os.environ.get("CLAUDE_PLUGIN_ROOT"):
        claude_reasons.append("CLAUDE_PLUGIN_ROOT")
    claude_code_keys = sorted([key for key in os.environ if key.startswith("CLAUDE_CODE_")])
    if claude_code_keys:
        claude_reasons.append(f"CLAUDE_CODE_*={len(claude_code_keys)}")
    if claude_reasons:
        return RuntimeDetection(
            provider="claude",
            confidence=signal_confidence(0.76, len(claude_reasons)),
            reasons=claude_reasons,
        )

    codex_reasons: list[str] = []
    codex_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    if os.environ.get("CODEX_HOME"):
        codex_reasons.append("CODEX_HOME")
    if (codex_home / "config.toml").exists():
        codex_reasons.append("~/.codex/config.toml")
    if codex_reasons:
        return RuntimeDetection(
            provider="codex",
            confidence=signal_confidence(0.72, len(codex_reasons)),
            reasons=codex_reasons,
        )

    opencode_reasons: list[str] = []
    for key in ("OPENCODE_CONFIG", "OPENCODE_CONFIG_DIR", "OPENCODE_CONFIG_CONTENT"):
        if os.environ.get(key):
            opencode_reasons.append(key)
    if (Path.home() / ".config" / "opencode" / "opencode.json").exists():
        opencode_reasons.append("~/.config/opencode/opencode.json")
    if (Path.home() / ".local" / "share" / "opencode").exists():
        opencode_reasons.append("~/.local/share/opencode")
    if opencode_reasons:
        return RuntimeDetection(
            provider="opencode",
            confidence=signal_confidence(0.7, len(opencode_reasons)),
            reasons=opencode_reasons,
        )

    return RuntimeDetection(provider="unknown", confidence=0.0, reasons=[f"no_runtime_signals:{repo_root.name}"])


def transcript_roots_for_cursor_repo(repo_root: Path) -> list[tuple[int, Path]]:
    projects_root = Path.home() / ".cursor" / "projects"
    if not projects_root.exists():
        return []

    slug = cursor_project_slug(repo_root).lower()
    repo_name = repo_root.name.lower()
    tokens = project_identifier_tokens(repo_root)
    results: list[tuple[int, Path]] = []

    for entry in projects_root.iterdir():
        if not entry.is_dir():
            continue

        transcript_dir = entry / "agent-transcripts"
        if not transcript_dir.exists():
            continue

        name = entry.name.lower()
        score = 0
        if name == slug:
            score += 1000
        if repo_name in name:
            score += 300
        for token in tokens:
            if token in name:
                score += 20

        if score > 0:
            results.append((score, transcript_dir))

    results.sort(key=lambda item: item[0], reverse=True)
    return results


def validate_explicit_transcript_path(explicit_transcript: str | None) -> Path | None:
    if explicit_transcript:
        path = Path(explicit_transcript).expanduser().resolve()
        if not path.exists():
            raise BranchContextError(f"transcript file does not exist: {path}")
        if path.suffix != ".jsonl":
            raise BranchContextError(f"transcript must be .jsonl: {path}")
        return path
    return None


def discover_cursor_transcript(repo_root: Path) -> Path | None:
    ranked_candidates: list[TranscriptCandidate] = []
    for score, transcript_root in transcript_roots_for_cursor_repo(repo_root):
        root_candidates = sorted(
            transcript_root.glob("**/*.jsonl"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not root_candidates:
            continue
        latest = root_candidates[0]
        ranked_candidates.append(
            TranscriptCandidate(path=latest, score=score, modified_at=latest.stat().st_mtime)
        )

    if not ranked_candidates:
        return None

    ranked_candidates.sort(key=lambda item: (item.score, item.modified_at), reverse=True)
    return ranked_candidates[0].path


def detect_claude_transcript_from_env() -> Path | None:
    env_candidates: list[TranscriptCandidate] = []
    for key, value in os.environ.items():
        if not value.strip():
            continue
        key_lower = key.lower()
        if "transcript" not in key_lower:
            continue
        if "path" not in key_lower and not value.lower().endswith(".jsonl"):
            continue

        candidate_path = Path(value).expanduser()
        if not candidate_path.exists() or candidate_path.suffix != ".jsonl":
            continue

        score = 50
        if "claude" in key_lower:
            score += 40
        if "hook" in key_lower:
            score += 10
        env_candidates.append(
            TranscriptCandidate(path=candidate_path.resolve(), score=score, modified_at=candidate_path.stat().st_mtime)
        )

    if not env_candidates:
        return None
    env_candidates.sort(key=lambda item: (item.score, item.modified_at), reverse=True)
    return env_candidates[0].path


def discover_claude_transcript(repo_root: Path) -> Path | None:
    from_env = detect_claude_transcript_from_env()
    if from_env:
        return from_env

    projects_root = Path.home() / ".claude" / "projects"
    if not projects_root.exists():
        return None

    tokens = project_identifier_tokens(repo_root)
    repo_name = repo_root.name.lower()
    scoped_candidates: list[TranscriptCandidate] = []

    direct_candidates = list(projects_root.glob("*/*.jsonl"))
    if not direct_candidates:
        direct_candidates = list(projects_root.glob("**/*.jsonl"))

    for path in direct_candidates:
        path_lower = path.as_posix().lower()
        score = 0
        if repo_name in path_lower:
            score += 8
        for token in tokens:
            if token in path_lower:
                score += 2
        if score <= 0:
            continue
        scoped_candidates.append(TranscriptCandidate(path=path.resolve(), score=score, modified_at=path.stat().st_mtime))

    if not scoped_candidates:
        return None

    scoped_candidates.sort(key=lambda item: (item.score, item.modified_at), reverse=True)
    return scoped_candidates[0].path


def infer_role(payload: dict[str, object]) -> str:
    direct_role = payload.get("role")
    if isinstance(direct_role, str) and direct_role.strip():
        role = direct_role.strip().lower()
    else:
        role = ""

    if not role:
        payload_type = payload.get("type")
        if isinstance(payload_type, str) and payload_type.lower() in KNOWN_CHAT_ROLES:
            role = payload_type.lower()

    if not role:
        message = payload.get("message")
        if isinstance(message, dict):
            nested_role = message.get("role")
            if isinstance(nested_role, str) and nested_role.strip():
                role = nested_role.strip().lower()

    if not role:
        author = payload.get("author")
        if isinstance(author, dict):
            author_role = author.get("role")
            if isinstance(author_role, str) and author_role.strip():
                role = author_role.strip().lower()

    if not role:
        sender = payload.get("sender")
        if isinstance(sender, str) and sender.strip():
            role = sender.strip().lower()

    role_aliases = {"human": "user", "ai": "assistant"}
    normalized = role_aliases.get(role, role)
    return normalized or "unknown"


def collect_text_fragments(value: object, depth: int = 0) -> list[str]:
    if depth > 5:
        return []

    if isinstance(value, str):
        return [value] if value.strip() else []

    if isinstance(value, list):
        list_fragments: list[str] = []
        for item in value:
            list_fragments.extend(collect_text_fragments(item, depth + 1))
        return list_fragments

    if not isinstance(value, dict):
        return []

    fragments: list[str] = []
    if value.get("type") == "text":
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            fragments.append(text)

    for key in ("text", "content", "body", "message", "prompt", "response", "input", "output", "value"):
        if key in value:
            fragments.extend(collect_text_fragments(value[key], depth + 1))
    return fragments


def parse_payload_to_message(payload: object) -> ChatMessage | None:
    if not isinstance(payload, dict):
        return None

    role = infer_role(payload)
    raw_chunks: list[str] = []
    cleaned_chunks: list[str] = []

    message = payload.get("message")
    content = message.get("content", []) if isinstance(message, dict) else []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            raw_chunks.append(text)
            cleaned = clean_text(text)
            if cleaned:
                cleaned_chunks.append(cleaned)

    if not cleaned_chunks:
        seen: set[str] = set()
        for text in collect_text_fragments(payload):
            cleaned = clean_text(text)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            raw_chunks.append(text)
            cleaned_chunks.append(cleaned)

    if not cleaned_chunks:
        return None
    return ChatMessage(role=role, text="\n\n".join(cleaned_chunks), raw_text="\n\n".join(raw_chunks))


def parse_json_payload_file(path: Path) -> list[object]:
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("messages", "conversation", "events", "entries", "items", "history"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    return []


def parse_transcript(transcript_path: Path) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    payloads: list[object] = []

    if transcript_path.suffix == ".jsonl":
        try:
            raw_lines = transcript_path.read_text().splitlines()
        except OSError:
            raw_lines = []
        for raw_line in raw_lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            payloads.append(payload)
    else:
        payloads = parse_json_payload_file(transcript_path)

    for payload in payloads:
        message = parse_payload_to_message(payload)
        if message:
            messages.append(message)
    return messages


def transcript_has_chat_messages(transcript_path: Path) -> bool:
    messages = parse_transcript(transcript_path)
    return any(message.role in KNOWN_CHAT_ROLES and len(message.text) >= 10 for message in messages)


def codex_home_dir() -> Path:
    return Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()


def discover_codex_transcript(_repo_root: Path) -> Path | None:
    home = codex_home_dir()
    if not home.exists():
        return None

    sessions_dir = home / "sessions"
    session_candidates: list[Path] = []
    if sessions_dir.exists():
        session_candidates = sorted(
            sessions_dir.glob("**/*.jsonl"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )

    for candidate in session_candidates:
        if transcript_has_chat_messages(candidate):
            return candidate.resolve()

    history_candidate = home / "history.jsonl"
    if history_candidate.exists() and transcript_has_chat_messages(history_candidate):
        return history_candidate.resolve()
    return None


def discover_opencode_transcript(repo_root: Path) -> Path | None:
    storage_root = Path.home() / ".local" / "share" / "opencode" / "project"
    if not storage_root.exists():
        return None

    filename_hint_regex = re.compile(r"(chat|session|transcript|message|history)", re.IGNORECASE)
    repo_tokens = [repo_root.name.lower(), *project_identifier_tokens(repo_root)]

    candidates: list[TranscriptCandidate] = []
    for pattern in ("**/*.jsonl", "**/*.json"):
        for path in storage_root.glob(pattern):
            if not path.is_file():
                continue
            path_text = path.as_posix().lower()
            score = 0
            for token in repo_tokens:
                if token in path_text:
                    score += 2
            if filename_hint_regex.search(path.name):
                score += 5
            candidates.append(TranscriptCandidate(path=path.resolve(), score=score, modified_at=path.stat().st_mtime))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item.score, item.modified_at), reverse=True)
    for candidate in candidates[:60]:
        if transcript_has_chat_messages(candidate.path):
            return candidate.path
    return None


def discover_transcript_for_runtime(repo_root: Path, runtime_provider: str) -> Path | None:
    if runtime_provider == "none":
        return None
    if runtime_provider == "cursor":
        return discover_cursor_transcript(repo_root)
    if runtime_provider == "claude":
        return discover_claude_transcript(repo_root)
    if runtime_provider == "codex":
        return discover_codex_transcript(repo_root)
    if runtime_provider == "opencode":
        return discover_opencode_transcript(repo_root)
    if runtime_provider == "unknown":
        for provider in ("cursor", "claude", "codex", "opencode"):
            candidate = discover_transcript_for_runtime(repo_root, provider)
            if candidate:
                return candidate
        return None
    return None


def discover_latest_transcript(
    repo_root: Path, explicit_transcript: str | None, runtime_provider: str
) -> Path | None:
    if runtime_provider == "none":
        return None

    explicit_path = validate_explicit_transcript_path(explicit_transcript)
    if explicit_path:
        return explicit_path

    return discover_transcript_for_runtime(repo_root, runtime_provider)


def extract_conversation_snapshot(messages: list[ChatMessage]) -> list[str]:
    tail = messages[-MAX_TRANSCRIPT_MESSAGES:]
    lines: list[str] = []
    for message in tail:
        short = truncate(clean_text(message.text).replace("\n", " "), MAX_MESSAGE_CHARS)
        lines.append(f"- {message.role}: {short}")
    return lines


def latest_user_objective(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role != "user":
            continue
        query_match = re.search(r"<user_query>(.*?)</user_query>", message.text, flags=re.DOTALL)
        if query_match:
            line = clean_text(query_match.group(1)).replace("\n", " ")
        else:
            line = clean_text(message.text).replace("\n", " ")
        if line:
            return truncate(line, 500)
    return "No user objective detected in transcript."


def extract_decisions(messages: list[ChatMessage]) -> list[str]:
    decisions: list[str] = []
    seen: set[str] = set()
    decision_regex = re.compile(r"Question (.*?): Selected option\(s\)\s*(.+)", re.IGNORECASE)

    for message in messages:
        for line in message.text.splitlines():
            match = decision_regex.search(line.strip())
            if not match:
                continue
            key = clean_text(match.group(1))
            value = clean_text(match.group(2))
            text = f"{key}: {value}"
            if text not in seen:
                seen.add(text)
                decisions.append(text)

    if decisions:
        return decisions

    fallback: list[str] = []
    for message in reversed(messages):
        if message.role != "user":
            continue
        cleaned = clean_text(message.text)
        if not cleaned:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
            sentence = sentence.strip()
            if len(sentence) < 25:
                continue
            if not re.search(r"\b(should|must|need|want|policy|decide|default)\b", sentence, flags=re.IGNORECASE):
                continue
            fallback.append(truncate(sentence, 180))
            if len(fallback) >= 4:
                return fallback
    return ["No explicit decisions parsed from transcript."]


def extract_path_candidates_from_messages(messages: list[ChatMessage]) -> set[Path]:
    path_regex = re.compile(r"(/[^\s'\"`]+\.plan\.md)")
    candidates: set[Path] = set()
    for message in messages:
        for match in path_regex.findall(message.raw_text):
            maybe = Path(match)
            if maybe.exists():
                candidates.add(maybe.resolve())
    return candidates


def branch_tokens(branch: str) -> list[str]:
    parts = re.split(r"[^a-zA-Z0-9]+", branch.lower())
    return [part for part in parts if len(part) >= 3]


def candidate_relevance_score(path: Path, tokens: list[str]) -> int:
    filename = path.name.lower()
    score = 0
    for token in tokens:
        if token in filename:
            score += 3

    if score > 0:
        return score

    try:
        preview = path.read_text()[:2000].lower()
    except UnicodeDecodeError:
        preview = ""

    for token in tokens:
        if token in preview:
            score += 1
    return score


def plan_candidate_paths(repo_root: Path, messages: list[ChatMessage], branch: str) -> list[Path]:
    referenced = [path.resolve() for path in extract_path_candidates_from_messages(messages) if path.exists()]
    referenced = sorted(set(referenced), key=lambda path: path.stat().st_mtime, reverse=True)
    if referenced:
        return referenced[:20]

    fallback_candidates: set[Path] = set()
    fallback_candidates.update((repo_root / ".cursor" / "plans").glob("*.plan.md"))

    tokens = branch_tokens(branch)
    scored: list[tuple[int, float, Path]] = []
    for path in fallback_candidates:
        if not path.exists():
            continue
        score = candidate_relevance_score(path, tokens)
        scored.append((score, path.stat().st_mtime, path.resolve()))

    if not scored:
        return []

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    positive = [item[2] for item in scored if item[0] > 0]
    if positive:
        return positive[:5]
    return [scored[0][2]]


def slugify(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = lowered.strip("-")
    return lowered or "plan"


def split_frontmatter(markdown: str) -> tuple[str | None, str]:
    if not markdown.startswith("---\n"):
        return None, markdown
    parts = markdown.split("\n---\n", 1)
    if len(parts) != 2:
        return None, markdown
    return parts[0][4:], parts[1]


def parse_frontmatter_name(frontmatter: str | None) -> str | None:
    if not frontmatter:
        return None
    for line in frontmatter.splitlines():
        stripped = line.strip()
        if not stripped.startswith("name:"):
            continue
        _, value = stripped.split(":", 1)
        cleaned = value.strip().strip("'\"")
        if cleaned:
            return cleaned
    return None


def parse_frontmatter_todos(frontmatter: str | None) -> list[PlanTodo]:
    if not frontmatter:
        return []

    todo_lines: list[str] = []
    in_todos = False
    for line in frontmatter.splitlines():
        if line.strip() == "todos:":
            in_todos = True
            continue
        if in_todos and line and not line.startswith("  "):
            break
        if in_todos:
            todo_lines.append(line)

    todos: list[PlanTodo] = []
    current: dict[str, str] = {}
    for line in todo_lines:
        stripped = line.strip()
        if stripped.startswith("- id:"):
            if current:
                todos.append(
                    PlanTodo(
                        item_id=current.get("id", "").strip(),
                        content=current.get("content", "").strip(),
                        status=current.get("status", "pending").strip().lower(),
                    )
                )
            current = {"id": stripped.split(":", 1)[1].strip()}
            continue
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if key not in {"content", "status"}:
            continue
        current[key] = raw_value.strip().strip("'\"")

    if current:
        todos.append(
            PlanTodo(
                item_id=current.get("id", "").strip(),
                content=current.get("content", "").strip(),
                status=current.get("status", "pending").strip().lower(),
            )
        )

    return [todo for todo in todos if todo.content]


def parse_plan_record(path: Path) -> PlanRecord:
    content = path.read_text()
    frontmatter, body = split_frontmatter(content)
    title = parse_frontmatter_name(frontmatter)
    if not title:
        heading_match = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
        title = heading_match.group(1).strip() if heading_match else path.stem

    todos = parse_frontmatter_todos(frontmatter)
    slug = slugify(title)
    return PlanRecord(
        slug=slug,
        source_path=path.resolve(),
        title=title,
        updated_at=path.stat().st_mtime,
        content=content,
        todos=todos,
    )


def select_latest_plans(plan_paths: list[Path]) -> dict[str, PlanRecord]:
    selected: dict[str, PlanRecord] = {}
    for path in plan_paths:
        record = parse_plan_record(path)
        existing = selected.get(record.slug)
        if not existing or record.updated_at > existing.updated_at:
            selected[record.slug] = record
    return selected


def resolve_base_ref(repo_root: Path) -> str | None:
    for candidate in ("origin/main", "main", "origin/master", "master"):
        proc = run_command(["git", "rev-parse", "--verify", candidate], cwd=repo_root, check=False)
        if proc.returncode == 0:
            return candidate
    return None


def working_set(repo_root: Path) -> tuple[list[str], list[str]]:
    status_proc = run_command(
        ["git", "status", "--short", "--untracked-files=all"], cwd=repo_root, check=False
    )
    status_lines = [line for line in status_proc.stdout.splitlines() if line.strip()]

    base_ref = resolve_base_ref(repo_root)
    if not base_ref:
        return status_lines, []

    diff_proc = run_command(["git", "diff", "--name-only", f"{base_ref}...HEAD"], cwd=repo_root, check=False)
    diff_lines = [line for line in diff_proc.stdout.splitlines() if line.strip()]
    return status_lines, diff_lines


def extract_marked_block(text: str, start_marker: str, end_marker: str, fallback: str) -> str:
    pattern = re.compile(re.escape(start_marker) + r"\n(.*?)\n" + re.escape(end_marker), re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip() or fallback
    return fallback


def write_notes(
    repo_root: Path,
    notes_path: Path,
    branch: str,
    branch_dir_key: str,
    transcript_path: Path | None,
    messages: list[ChatMessage],
    decisions: list[str],
    status_lines: list[str],
    changed_files: list[str],
    plan_records: dict[str, PlanRecord],
) -> None:
    previous = notes_path.read_text() if notes_path.exists() else ""
    manual_notes = extract_marked_block(previous, MANUAL_NOTES_START, MANUAL_NOTES_END, DEFAULT_MANUAL_NOTE)
    conversation_snapshot = extract_conversation_snapshot(messages)

    status_block = status_lines if status_lines else ["- clean working tree"]
    changed_block = [f"- `{item}`" for item in changed_files] if changed_files else ["- none"]
    plan_block = [
        f"- `{slug}.md` from `{display_path(record.source_path, repo_root)}`"
        for slug, record in sorted(plan_records.items())
    ]
    if not plan_block:
        plan_block = ["- none"]

    notes_content = "\n".join(
        [
            f"# Branch Context: {branch}",
            "",
            "## Metadata (generated)",
            f"- branch: `{branch}`",
            f"- branch_key: `{branch_dir_key}`",
            f"- updated_at_utc: `{dt.datetime.now(dt.UTC).isoformat()}`",
            f"- transcript_source: `{display_transcript_source(transcript_path, repo_root)}`",
            f"- captured_plan_files: `{len(plan_records)}`",
            "",
            "## Active Objective (generated)",
            latest_user_objective(messages),
            "",
            "## Key Decisions (generated)",
            *[f"- {item}" for item in decisions],
            "",
            "## Working Set (generated)",
            "### Git Status",
            *status_block,
            "",
            "### Changed Files Against Base",
            *changed_block,
            "",
            "## Plan References (generated)",
            *plan_block,
            "",
            "## Recent Conversation Snapshot (generated)",
            *(conversation_snapshot or ["- no transcript messages captured"]),
            "",
            "## Manual Notes",
            MANUAL_NOTES_START,
            manual_notes,
            MANUAL_NOTES_END,
            "",
        ]
    )
    notes_path.write_text(notes_content)


def classify_todos(plan_records: dict[str, PlanRecord]) -> tuple[list[str], list[str], list[str]]:
    active: list[str] = []
    blocked: list[str] = []
    done: list[str] = []
    seen: set[str] = set()

    for slug, record in sorted(plan_records.items()):
        for todo in record.todos:
            item_text = f"[{slug}] {todo.content}"
            if item_text in seen:
                continue
            seen.add(item_text)

            status = todo.status.lower()
            if status in {"completed", "done"}:
                done.append(item_text)
            elif status == "blocked":
                blocked.append(item_text)
            elif status in {"cancelled", "canceled"}:
                done.append(f"[cancelled] {item_text}")
            else:
                active.append(item_text)

    return active, blocked, done


def write_todo(todo_path: Path, branch: str, plan_records: dict[str, PlanRecord]) -> None:
    previous = todo_path.read_text() if todo_path.exists() else ""
    manual_todo = extract_marked_block(previous, MANUAL_TODO_START, MANUAL_TODO_END, DEFAULT_MANUAL_TODO)
    active, blocked, done = classify_todos(plan_records)

    active_lines = [f"- [ ] {item}" for item in active] or ["- none"]
    blocked_lines = [f"- [ ] {item}" for item in blocked] or ["- none"]
    done_lines = [f"- [x] {item}" for item in done] or ["- none"]

    todo_content = "\n".join(
        [
            f"# Branch Todo: {branch}",
            "",
            "## Active (generated)",
            *active_lines,
            "",
            "## Blocked (generated)",
            *blocked_lines,
            "",
            "## Done (generated)",
            *done_lines,
            "",
            "## Manual Todo",
            MANUAL_TODO_START,
            manual_todo,
            MANUAL_TODO_END,
            "",
        ]
    )
    todo_path.write_text(todo_content)


def sync_plans(plans_dir: Path, plan_records: dict[str, PlanRecord]) -> None:
    plans_dir.mkdir(parents=True, exist_ok=True)
    expected: set[str] = set()
    for slug, record in plan_records.items():
        target = plans_dir / f"{slug}.md"
        target.write_text(record.content)
        expected.add(target.name)

    for existing in plans_dir.glob("*.md"):
        if existing.name not in expected:
            existing.unlink()


def stage_and_commit_context(
    repo_root: Path, pathspecs: list[str], commit_message: str
) -> tuple[str | None, bool]:
    if not pathspecs:
        raise BranchContextError("no pathspecs provided for context commit")

    add_proc = run_command(["git", "add", "-A", "-f", "--", *pathspecs], cwd=repo_root, check=False)
    if add_proc.returncode != 0:
        stderr = (add_proc.stderr or "").lower()
        stdout = (add_proc.stdout or "").lower()
        if "did not match any files" not in stderr and "did not match any files" not in stdout:
            detail = add_proc.stderr.strip() or add_proc.stdout.strip() or "git add failed"
            raise BranchContextError(detail)
    diff_proc = run_command(["git", "diff", "--cached", "--quiet", "--", *pathspecs], cwd=repo_root, check=False)
    if diff_proc.returncode == 0:
        return None, False

    run_command(["git", "commit", "-m", commit_message], cwd=repo_root)
    head = run_command(["git", "rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
    return head, True


def context_paths(repo_root: Path, key: str) -> tuple[Path, Path]:
    current = repo_root / CONTEXT_ROOT_DIR / key
    legacy = repo_root / LEGACY_CONTEXT_ROOT_DIR / key
    return current, legacy


def migrate_legacy_context(repo_root: Path, key: str) -> tuple[Path, bool]:
    current, legacy = context_paths(repo_root, key)
    if legacy.exists() and not current.exists():
        current.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(current))
        return current, True
    return current, False


def update_context(
    repo_root: Path,
    branch: str,
    transcript_override: str | None,
    runtime_override: str | None,
    commit_changes: bool,
) -> int:
    key = branch_key(branch)
    context_dir, migrated_legacy = migrate_legacy_context(repo_root, key)
    plans_dir = context_dir / "plans"
    notes_path = context_dir / "notes.md"
    todo_path = context_dir / "todo.md"

    context_dir.mkdir(parents=True, exist_ok=True)
    runtime_detection = detect_runtime(repo_root, runtime_override)
    transcript_path = discover_latest_transcript(
        repo_root=repo_root,
        explicit_transcript=transcript_override,
        runtime_provider=runtime_detection.provider,
    )
    messages: list[ChatMessage] = parse_transcript(transcript_path) if transcript_path else []
    decisions = extract_decisions(messages)
    plan_paths = plan_candidate_paths(repo_root, messages, branch)
    latest_plans = select_latest_plans(plan_paths)
    status_lines, changed_files = working_set(repo_root)

    sync_plans(plans_dir, latest_plans)
    write_notes(
        repo_root=repo_root,
        notes_path=notes_path,
        branch=branch,
        branch_dir_key=key,
        transcript_path=transcript_path,
        messages=messages,
        decisions=decisions,
        status_lines=status_lines,
        changed_files=changed_files,
        plan_records=latest_plans,
    )
    write_todo(todo_path=todo_path, branch=branch, plan_records=latest_plans)

    commit_hash: str | None = None
    committed = False
    if commit_changes:
        pathspecs = [f"{CONTEXT_ROOT_DIR}/{key}"]
        if migrated_legacy or (repo_root / LEGACY_CONTEXT_ROOT_DIR / key).exists():
            pathspecs.append(f"{LEGACY_CONTEXT_ROOT_DIR}/{key}")
        commit_hash, committed = stage_and_commit_context(
            repo_root=repo_root,
            pathspecs=pathspecs,
            commit_message=f"chore(context): update branch context for {branch}",
        )

    print(f"context_dir={display_path(context_dir, repo_root)}")
    print(f"runtime={runtime_detection.provider}")
    print(f"runtime_confidence={runtime_detection.confidence:.2f}")
    print(f"runtime_reasons={','.join(runtime_detection.reasons)}")
    print(f"transcript_source={display_transcript_source(transcript_path, repo_root)}")
    print(f"captured_plans={len(latest_plans)}")
    if commit_changes:
        if committed:
            print(f"commit_hash={commit_hash}")
        else:
            print("commit_hash=none (no context changes)")
    return 0


def extract_section(markdown: str, heading: str) -> str:
    pattern = re.compile(
        rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(markdown)
    if not match:
        return ""
    return match.group(1).strip()


def parse_checkbox_items(section: str) -> list[str]:
    items: list[str] = []
    for line in section.splitlines():
        match = re.match(r"^\s*-\s*\[[ xX]\]\s*(.+)$", line)
        if match:
            items.append(match.group(1).strip())
    return items


def load_context_payload(repo_root: Path, branch: str) -> dict[str, object]:
    key = branch_key(branch)
    current, legacy = context_paths(repo_root, key)
    context_dir = current if current.exists() else legacy
    notes_path = context_dir / "notes.md"
    todo_path = context_dir / "todo.md"
    plans_dir = context_dir / "plans"

    missing: list[str] = []
    if not notes_path.exists():
        missing.append(str(notes_path))
    if not todo_path.exists():
        missing.append(str(todo_path))
    if not plans_dir.exists():
        missing.append(str(plans_dir))
    if missing:
        raise BranchContextError(
            "missing branch context files. run `scripts/branch_context_manager.sh update` first.\n"
            + "\n".join(f"- {item}" for item in missing)
        )

    notes = notes_path.read_text()
    todo = todo_path.read_text()
    plans = sorted(plans_dir.glob("*.md"))
    if not plans:
        raise BranchContextError(f"no plans found in {plans_dir}")

    objective = extract_section(notes, "Active Objective (generated)")
    decisions_section = extract_section(notes, "Key Decisions (generated)")
    decisions = [re.sub(r"^\s*-\s*", "", line).strip() for line in decisions_section.splitlines() if line.strip().startswith("-")]

    active = parse_checkbox_items(extract_section(todo, "Active (generated)"))
    blocked = parse_checkbox_items(extract_section(todo, "Blocked (generated)"))
    done = parse_checkbox_items(extract_section(todo, "Done (generated)"))

    plan_summaries = []
    for plan in plans:
        content = plan.read_text()
        heading_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
        title = heading_match.group(1).strip() if heading_match else plan.stem
        plan_summaries.append(
            {
                "slug": plan.stem,
                "title": title,
                "path": str(plan),
                "updated_at": dt.datetime.fromtimestamp(plan.stat().st_mtime, dt.UTC).isoformat(),
            }
        )

    return {
        "branch": branch,
        "branch_key": key,
        "context_dir": str(context_dir),
        "notes_path": str(notes_path),
        "todo_path": str(todo_path),
        "plans_path": str(plans_dir),
        "summary": {
            "objective": objective,
            "decisions": decisions,
            "active_todos": active,
            "blocked_todos": blocked,
            "done_todos": done,
            "plan_count": len(plan_summaries),
        },
        "plans": plan_summaries,
    }


def print_load_summary(payload: dict[str, object]) -> None:
    summary_obj = payload.get("summary")
    summary = summary_obj if isinstance(summary_obj, dict) else {}
    plans_obj = payload.get("plans")
    plans = plans_obj if isinstance(plans_obj, list) else []

    objective_obj = summary.get("objective")
    objective = objective_obj if isinstance(objective_obj, str) and objective_obj.strip() else "none"
    print(f"# Loaded Branch Context: {payload.get('branch')}")
    print("")
    print("## Objective")
    print(objective)
    print("")

    def print_items(title: str, values: Iterable[str]) -> None:
        items = list(values)
        print(f"## {title}")
        if not items:
            print("- none")
        else:
            for item in items:
                print(f"- {item}")
        print("")

    def str_items(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    print_items("Decisions", str_items(summary.get("decisions")))
    print_items("Active Todos", str_items(summary.get("active_todos")))
    print_items("Blocked Todos", str_items(summary.get("blocked_todos")))
    print_items("Done Todos", str_items(summary.get("done_todos")))

    print("## Plans")
    if not plans:
        print("- none")
    else:
        for plan in plans:
            if not isinstance(plan, dict):
                continue
            slug = plan.get("slug")
            title = plan.get("title")
            path = plan.get("path")
            slug_str = slug if isinstance(slug, str) and slug else "unknown"
            title_str = title if isinstance(title, str) and title else "unknown"
            path_str = path if isinstance(path, str) and path else "unknown"
            print(f"- `{slug_str}`: {title_str} ({path_str})")


def delete_context(repo_root: Path, branch: str, commit_changes: bool) -> int:
    key = branch_key(branch)
    current, legacy = context_paths(repo_root, key)
    deleted: list[Path] = []
    for path in (current, legacy):
        if path.exists():
            shutil.rmtree(path)
            deleted.append(path)

    if not deleted:
        raise BranchContextError(f"branch context does not exist: {current}")

    for path in deleted:
        print(f"deleted={display_path(path, repo_root)}")

    commit_hash: str | None = None
    committed = False
    if commit_changes:
        pathspecs: list[str] = [f"{CONTEXT_ROOT_DIR}/{key}"]
        if (repo_root / LEGACY_CONTEXT_ROOT_DIR / key).exists():
            pathspecs.append(f"{LEGACY_CONTEXT_ROOT_DIR}/{key}")
        commit_hash, committed = stage_and_commit_context(
            repo_root=repo_root,
            pathspecs=pathspecs,
            commit_message=f"chore(context): delete branch context for {branch}",
        )

    if commit_changes:
        if committed:
            print(f"commit_hash={commit_hash}")
        else:
            print("commit_hash=none (no staged context changes)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Manage branch-scoped context files in {CONTEXT_ROOT_DIR}/<branch-key>/"
    )
    parser.add_argument("--repo-root", help="Optional git repo root path")
    parser.add_argument("--branch", help="Optional branch name override")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    update_parser = subparsers.add_parser("update", help="Update branch context from git + transcript")
    update_parser.add_argument("--transcript-path", help="Optional explicit transcript .jsonl path")
    update_parser.add_argument(
        "--runtime",
        choices=SUPPORTED_RUNTIME_OVERRIDES,
        help="Runtime override for transcript discovery (default: auto or BRANCH_CONTEXT_RUNTIME)",
    )
    update_parser.add_argument("--no-commit", action="store_true", help="Write files without creating a commit")

    load_parser = subparsers.add_parser("load", help="Load branch context")
    load_parser.add_argument("--json", action="store_true", help="Output structured JSON payload")

    delete_parser = subparsers.add_parser("delete", help="Delete branch context")
    delete_parser.add_argument("--no-commit", action="store_true", help="Delete files without creating a commit")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        repo_root = git_repo_root(args.repo_root)
        branch = current_branch(repo_root, args.branch)

        if args.mode == "update":
            return update_context(
                repo_root=repo_root,
                branch=branch,
                transcript_override=args.transcript_path,
                runtime_override=args.runtime,
                commit_changes=not args.no_commit,
            )

        if args.mode == "load":
            payload = load_context_payload(repo_root=repo_root, branch=branch)
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print_load_summary(payload)
            return 0

        if args.mode == "delete":
            return delete_context(
                repo_root=repo_root,
                branch=branch,
                commit_changes=not args.no_commit,
            )

        raise BranchContextError(f"unsupported mode: {args.mode}")
    except BranchContextError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
