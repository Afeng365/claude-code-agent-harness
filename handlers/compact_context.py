import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from dao.anthropic_utils import client
from settings.constant import TOOL_RESULTS_DIR, WORKDIR, PERSISTED_OPEN, PERSISTED_PREVIEW_CHARS, PERSISTED_CLOSE, \
    KEEP_RECENT_TOOL_RESULTS, TRANSCRIPT_DIR, MODEL, PERSIST_OUTPUT_TRIGGER_CHARS_DEFAULT


# === persisted_output ===
@dataclass
class CompactState:
    has_compacted: bool = False
    last_summary: str = ""
    recent_files: list[str] = field(default_factory=list)


def estimate_context_size(messages: list) -> int:
    return len(str(messages))


def _persist_tool_result(tool_use_id: str, content: str) -> Path:
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", tool_use_id or "unknown")
    path = TOOL_RESULTS_DIR / f"{safe_id}.txt"
    if not path.exists():
        path.write_text(content)
    return path.relative_to(WORKDIR)


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def _preview_slice(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    idx = text[:limit].rfind("\n")
    cut = idx if idx > (limit * 0.) else limit
    return text[:cut], True


def _build_persisted_marker(stored_path: Path, content: str) -> str:
    preview, has_more = _preview_slice(content, PERSISTED_PREVIEW_CHARS)
    marker = (
        f"{PERSISTED_OPEN}\n"
        f"Output too large ({_format_size(len(content))}). "
        f"Full output saved to: {stored_path}\n\n"
        f"Preview (first {_format_size(PERSISTED_PREVIEW_CHARS)}):\n"
        f"{preview}"
    )
    if has_more:
        marker += "\n..."
    marker += f"\n{PERSISTED_CLOSE}"
    return marker


def maybe_persist_output(tool_use_id: str, output: str, trigger_chars: int = None) -> str:
    if not isinstance(output, str):
        return str(output)
    trigger = PERSIST_OUTPUT_TRIGGER_CHARS_DEFAULT if trigger_chars is None else int(trigger_chars)
    if len(output) <= trigger:
        return output
    stored_path = _persist_tool_result(tool_use_id, output)
    return _build_persisted_marker(stored_path, output)


def collect_tool_result_blocks(messages: list) -> list[tuple[int, int, dict]]:
    blocks = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append((message_index, block_index, block))
    return blocks


def micro_compact(messages: list) -> list:
    tool_results = collect_tool_result_blocks(messages)
    if len(tool_results) <= KEEP_RECENT_TOOL_RESULTS:
        return messages

    for _, _, block in tool_results[:-KEEP_RECENT_TOOL_RESULTS]:
        content = block.get("content", "")
        if not isinstance(content, str) or len(content) <= 120:
            continue
        block["content"] = "[Earlier tool result compacted. Re-run the tool if you need full detail.]"
    return messages


def write_transcript(messages: list) -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as handle:
        for message in messages:
            handle.write(json.dumps(message, default=str) + "\n")
    return path


def summarize_history(messages: list) -> str:
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = (
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve:\n"
        "1. The current goal\n"
        "2. Important findings and decisions\n"
        "3. Files read or changed\n"
        "4. Remaining work\n"
        "5. User constraints and preferences\n"
        "Be compact but concrete.\n\n"
        f"{conversation}"
    )
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    return response.content[0].text.strip()


def compact_history(messages: list, state: CompactState, focus: str | None = None) -> list:
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")

    summary = summarize_history(messages)
    if focus:
        summary += f"\n\nFocus to preserve next: {focus}"
    if state.recent_files:
        recent_lines = "\n".join(f"- {path}" for path in state.recent_files)
        summary += f"\n\nRecent files to reopen if needed:\n{recent_lines}"

    state.has_compacted = True
    state.last_summary = summary

    return [{
        "role": "user",
        "content": (
            "This conversation was compacted so the agent can continue working.\n\n"
            f"{summary}"
        ),
    }]