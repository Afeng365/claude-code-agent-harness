#!/usr/bin/env python3
# Harness: autonomy -- models that find work without being told.

import json
import platform
import os
import re
import random
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from queue import Queue, Empty
from datetime import datetime, timedelta
from anthropic import Anthropic, APIError
from bandit import plugins

from log import logging

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

PLAN_REMINDER_INTERVAL = 3

# -- Permission modes --
# Teaching version starts with three clear modes first.
MODES = ("default", "plan", "auto")

READ_ONLY_TOOLS = {"read_file", "bash_readonly"}

# Tools that modify state
WRITE_TOOLS = {"write_file", "edit_file", "bash"}

HOOK_EVENTS = ("PreToolUse", "PostToolUse", "SessionStart")
HOOK_TIMEOUT = 30  # seconds
TRUST_MARKER = WORKDIR / ".claude" / ".claude_trusted"

MEMORY_DIR = WORKDIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
MEMORY_TYPES = ("user", "feedback", "project", "reference")
MAX_INDEX_LINES = 200
DYNAMIC_BOUNDARY = "=== DYNAMIC_BOUNDARY ==="

TASKS_DIR = WORKDIR / ".tasks"
SKILLS_DIR = WORKDIR / "skills"

RUNTIME_DIR = WORKDIR / ".runtime-tasks"
RUNTIME_DIR.mkdir(exist_ok=True)
STALL_THRESHOLD_S = 45  # seconds before a task is considered stalled

# Recovery constants
MAX_RECOVERY_ATTEMPTS = 3
BACKOFF_BASE_DELAY = 1.0  # seconds
BACKOFF_MAX_DELAY = 30.0  # seconds
TOKEN_THRESHOLD = 50000  # chars / 4 ~ tokens for compact trigger

SCHEDULED_TASKS_FILE = WORKDIR / ".claude" / "scheduled_tasks.json"
CRON_LOCK_FILE = WORKDIR / ".claude" / "cron.lock"
AUTO_EXPIRY_DAYS = 7
JITTER_MINUTES = [0, 30]  # avoid these exact minutes for recurring tasks
JITTER_OFFSET_MAX = 4  # offset range in minutes
# Teaching version: use a simple 1-4 minute offset when needed.


# Team constants
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
REQUESTS_DIR = TEAM_DIR / "requests"
CLAIM_EVENTS_PATH = TASKS_DIR / "claim_events.jsonl"

POLL_INTERVAL = 5
IDLE_TIMEOUT = 300

CONTINUATION_MESSAGE = (
    "Output limit hit. Continue directly from where you stopped -- "
    "no recap, no repetition. Pick up mid-sentence if needed."
)

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval",
    "plan_approval_response",
}

PERMISSION_MODES = ("default", "auto")

_claim_lock = threading.Lock()


def detect_repo_root(cwd: Path) -> Path | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd, capture_output=True, check=True, text=True, timeout=10
        )
        root = Path(r.stdout.strip())
        return root if r.returncode == 0 and root.exists() else None
    except Exception as e:
        logging.error(f"error: {e}")
        return None


REPO_ROOT = detect_repo_root(WORKDIR) or WORKDIR


@dataclass
class SkillManifest:
    name: str
    description: str
    path: Path


@dataclass
class SkillDocument:
    manifest: SkillManifest
    body: str


class SkillRegistry:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.documents: dict[str, SkillDocument] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.skills_dir.exists():
            return

        for path in sorted(self.skills_dir.rglob("SKILL.md")):
            meta, body = self._parse_frontmatter_skill(path.read_text())
            name = meta.get("name", path.parent.name)
            description = meta.get("description", "No description")
            manifest = SkillManifest(name=name, description=description, path=path)
            self.documents[name] = SkillDocument(manifest=manifest, body=body.strip())

    def _parse_frontmatter_skill(self, text: str) -> tuple[dict, str]:
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text

        meta = {}
        for line in match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
        return meta, match.group(2)

    def describe_available(self) -> str:
        if not self.documents:
            return "No skills available"
        lines = []
        for name in sorted(self.documents):
            manifest = self.documents[name].manifest
            lines.append(f"- {manifest.name}: {manifest.description}")
        return "\n".join(lines)

    def load_full_text(self, name: str) -> str:
        document = self.documents.get(name)
        if not document:
            known = ", ".join(sorted(self.documents.keys())) or "(none)"
            return f"Error: Unknown skill '{name}'. Available skills: {known}"
        return (
            f"<skill name=\"{document.manifest.name}\">\n"
            f"{document.body}\n"
            "</skill>"
        )


SKILL_REGISTRY = SkillRegistry(SKILLS_DIR)


@dataclass
class PlanItem:
    content: str
    status: str = "pending"
    active_form: str = ""


@dataclass
class PlanningState:
    items: list[PlanItem] = field(default_factory=list)
    rounds_since_update: int = 0


class TodoManager:
    def __init__(self):
        self.state = PlanningState()

    def update(self, items: list):
        if len(items) > 12:
            raise ValueError("TodoManager can only update at most 12 items")
        normalized = []
        in_progress_count = 0
        for index, raw_item in enumerate(items):
            content = str(raw_item.get("content", "")).strip()
            status = str(raw_item.get("status", "pending")).lower()
            active_form = str(raw_item.get("active_form", "")).strip()

            if not content:
                raise ValueError(f"Item {index}: content required")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"Item {index}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1

            normalized.append(PlanItem(content=content, status=status, active_form=active_form))
        if in_progress_count > 1:
            raise ValueError("Only one plan item can be in_progress")

        self.state.items = normalized
        self.state.rounds_since_update = 0
        return self.render()

    def note_round_without_update(self) -> None:
        self.state.rounds_since_update += 1

    def reminder(self) -> str | None:
        if not self.state.items:
            return None
        if self.state.rounds_since_update < PLAN_REMINDER_INTERVAL:
            return None
        return "<reminder>Refresh your current plan before continuing.</reminder>"

    def render(self) -> str:
        if not self.state.items:
            return "No session plan yet"

        lines = []
        for item in self.state.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item.status]
            line = f"{marker} {item.content}"
            if item.status == "in_progress" and item.active_form:
                line += f" ({item.active_form})"
            lines.append(line)
        completed = sum(1 for item in self.state.items if item.status == "completed")
        lines.append(f"\n({completed}/{len(self.state.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


class BashSecurityValidator:
    """
    Validate bash commands for obviously dangerous patterns.

    The teaching version deliberately keeps this small and easy to read.
    First catch a few high-risk patterns, then let the permission pipeline
    decide whether to deny or ask the user.
    """

    VALIDATORS = [
        ("shell_metachar", r"[;&|`$]"),  # shell metacharacters
        ("sudo", r"\bsudo\b"),  # privilege escalation
        ("rm_rf", r"\brm\s+(-[a-zA-Z]*)?r"),  # recursive delete
        ("cmd_substitution", r"\$\("),  # command substitution
        ("ifs_injection", r"\bIFS\s*="),  # IFS manipulation
    ]

    def validate(self, command: str) -> list:
        """
        Check a bash command against all validators.

        Returns list of (validator_name, matched_pattern) tuples for failures.
        An empty list means the command passed all validators.
        """
        failures = []
        for name, pattern in self.VALIDATORS:
            if re.search(pattern, command):
                failures.append((name, pattern))
        return failures

    def is_safe(self, command: str) -> bool:
        """Convenience: returns True only if no validators triggered."""
        return len(self.validate(command)) == 0

    def describe_failures(self, command: str) -> str:
        """Human-readable summary of validation failures."""
        failures = self.validate(command)
        if not failures:
            return "No issues detected"
        parts = [f"{name} (pattern: {pattern})" for name, pattern in failures]
        return "Security flags: " + ", ".join(parts)


def is_workspace_trusted(workspace: Path = None) -> bool:
    """
    Check if a workspace has been explicitly marked as trusted.

    The teaching version uses a simple marker file. A more complete system
    can layer richer trust flows on top of the same idea.
    """
    ws = workspace or WORKDIR
    trust_marker = ws / ".claude" / ".claude_trusted"
    return trust_marker.exists()


# Singleton validator instance used by the permission pipeline
bash_validator = BashSecurityValidator()

# -- Permission rules --
# Rules are checked in order: first match wins.
# Format: {"tool": "<tool_name_or_*>", "path": "<glob_or_*>", "behavior": "allow|deny|ask"}
DEFAULT_RULES = [
    # Always deny dangerous patterns
    {"tool": "bash", "content": "rm -rf /", "behavior": "deny"},
    {"tool": "bash", "content": "sudo *", "behavior": "deny"},
    # Allow reading anything
    {"tool": "read_file", "path": "*", "behavior": "allow"},
]


class PermissionManager:
    """
    Manages permission decisions for tool calls.

    Pipeline: deny_rules -> mode_check -> allow_rules -> ask_user

    The teaching version keeps the decision path short on purpose so readers
    can implement it themselves before adding more advanced policy layers.
    """

    def __init__(self, mode: str = "default", rules: list = None):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}. Choose from {MODES}")
        self.mode = mode
        self.rules = rules or list(DEFAULT_RULES)
        # Simple denial tracking helps surface when the agent is repeatedly
        # asking for actions the system will not allow.
        self.consecutive_denials = 0
        self.max_consecutive_denials = 3

    def check(self, tool_name: str, tool_input: dict) -> dict:
        """
        Returns: {"behavior": "allow"|"deny"|"ask", "reason": str}
        """
        # Step 0: Bash security validation (before deny rules)
        # Teaching version checks early for clarity.
        if tool_name == "bash":
            command = tool_input.get("command", "")
            failures = bash_validator.validate(command)
            if failures:
                # Severe patterns (sudo, rm_rf) get immediate deny
                severe = {"sudo", "rm_rf"}
                severe_hits = [f for f in failures if f[0] in severe]
                if severe_hits:
                    desc = bash_validator.describe_failures(command)
                    return {"behavior": "deny",
                            "reason": f"Bash validator: {desc}"}
                # Other patterns escalate to ask (user can still approve)
                desc = bash_validator.describe_failures(command)
                return {"behavior": "allow",
                        "reason": f"Bash validator flagged: {desc}"}
        #
        # # Step 1: Deny rules (bypass-immune, checked first always)
        # for rule in self.rules:
        #     if rule["behavior"] != "deny":
        #         continue
        #     if self._matches(rule, tool_name, tool_input):
        #         return {"behavior": "deny",
        #                 "reason": f"Blocked by deny rule: {rule}"}
        #
        # # Step 2: Mode-based decisions
        # if self.mode == "plan":
        #     # Plan mode: deny all write operations, allow reads
        #     if tool_name in WRITE_TOOLS:
        #         return {"behavior": "deny",
        #                 "reason": "Plan mode: write operations are blocked"}
        #     return {"behavior": "allow", "reason": "Plan mode: read-only allowed"}
        #
        # if self.mode == "auto":
        #     # Auto mode: auto-allow read-only tools, ask for writes
        #     if tool_name in READ_ONLY_TOOLS or tool_name == "read_file":
        #         return {"behavior": "allow",
        #                 "reason": "Auto mode: read-only tool auto-approved"}
        #     # Teaching: fall through to allow rules, then ask
        #     pass
        #
        # # Step 3: Allow rules
        # for rule in self.rules:
        #     if rule["behavior"] != "allow":
        #         continue
        #     if self._matches(rule, tool_name, tool_input):
        #         self.consecutive_denials = 0
        #         return {"behavior": "allow",
        #                 "reason": f"Matched allow rule: {rule}"}

        # Step 4: Ask user (default behavior for unmatched tools)
        # return {"behavior": "ask",
        #         "reason": f"No rule matched for {tool_name}, asking user"}
        return {"behavior": "allow",
                "reason": "Auto mode: tool auto-approved"}

    def ask_user(self, tool_name: str, tool_input: dict) -> bool:
        """Interactive approval prompt. Returns True if approved."""
        preview = json.dumps(tool_input, ensure_ascii=False)[:200]
        logging.info(f"  [Permission] {tool_name}: {preview}")
        try:
            answer = input("  Allow? (y/n/always): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False

        if answer == "always":
            # Add permanent allow rule for this tool
            self.rules.append({"tool": tool_name, "path": "*", "behavior": "allow"})
            self.consecutive_denials = 0
            return True
        if answer in ("y", "yes"):
            self.consecutive_denials = 0
            return True

        # Track denials for circuit breaker
        self.consecutive_denials += 1
        if self.consecutive_denials >= self.max_consecutive_denials:
            logging.info(f"  [{self.consecutive_denials} consecutive denials -- "
                         "consider switching to plan mode]")
        return False

    def _matches(self, rule: dict, tool_name: str, tool_input: dict) -> bool:
        """Check if a rule matches the tool call."""
        # Tool name match
        if rule.get("tool") and rule["tool"] != "*":
            if rule["tool"] != tool_name:
                return False
        # Path pattern match
        if "path" in rule and rule["path"] != "*":
            path = tool_input.get("path", "")
            if not fnmatch(path, rule["path"]):
                return False
        # Content pattern match (for bash commands)
        if "content" in rule:
            command = tool_input.get("command", "")
            if not fnmatch(command, rule["content"]):
                return False
        return True


class HookManager:
    """
    Load and execute hooks from .hooks.json configuration.

    The hook manager does three simple jobs:
    - load hook definitions
    - run matching commands for an event
    - aggregate block / message results for the caller
    """

    def __init__(self, config_path: Path = None, sdk_mode: bool = False):
        self.hooks = {"PreToolUse": [], "PostToolUse": [], "SessionStart": []}
        self._sdk_mode = sdk_mode
        config_path = config_path or (WORKDIR / ".hooks.json")
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                for event in HOOK_EVENTS:
                    self.hooks[event] = config.get("hooks", {}).get(event, [])
                logging.info(f"[Hooks loaded from {config_path}]")
            except Exception as e:
                logging.error(f"[Hook config error: {e}]")

    def _check_workspace_trust(self) -> bool:
        """
        Check whether the current workspace is trusted.

        The teaching version uses a simple trust marker file.
        In SDK mode, trust is treated as implicit.
        """
        if self._sdk_mode:
            return True
        return TRUST_MARKER.exists()

    def run_hooks(self, event: str, context: dict = None) -> dict:
        """
        Execute all hooks for an event.

        Returns: {"blocked": bool, "messages": list[str]}
          - blocked: True if any hook returned exit code 1
          - messages: stderr content from exit-code-2 hooks (to inject)
        """
        # logging.info(f"[Running hooks for {event}], context={context}")
        result = {"blocked": False, "messages": []}

        # Trust gate: refuse to run hooks in untrusted workspaces
        if not self._check_workspace_trust():
            return result

        hooks = self.hooks.get(event, [])

        for hook_def in hooks:
            # Check matcher (tool name filter for PreToolUse/PostToolUse)
            matcher = hook_def.get("matcher")
            if matcher and context:
                tool_name = context.get("tool_name", "")
                if matcher != "*" and matcher != tool_name:
                    continue

            command = hook_def.get("command", "")
            if not command:
                continue

            # Build environment with hook context
            env = dict(os.environ)
            if context:
                env["HOOK_EVENT"] = event
                env["HOOK_TOOL_NAME"] = context.get("tool_name", "")
                env["HOOK_TOOL_INPUT"] = json.dumps(
                    context.get("tool_input", {}), ensure_ascii=False)[:10000]
                if "tool_output" in context:
                    env["HOOK_TOOL_OUTPUT"] = str(
                        context["tool_output"])[:10000]

            try:
                r = subprocess.run(
                    command, shell=True, cwd=WORKDIR, env=env,
                    capture_output=True, text=True, timeout=HOOK_TIMEOUT,
                )

                if r.returncode == 0:
                    # Continue silently
                    if r.stdout.strip():
                        logging.info(f"  [hook:{event}] {r.stdout.strip()[:100]}")

                    # Optional structured stdout: small extension point that
                    # keeps the teaching contract simple.
                    try:
                        hook_output = json.loads(r.stdout)
                        if "updatedInput" in hook_output and context:
                            context["tool_input"] = hook_output["updatedInput"]
                        if "additionalContext" in hook_output:
                            result["messages"].append(
                                hook_output["additionalContext"])
                        if "permissionDecision" in hook_output:
                            result["permission_override"] = (
                                hook_output["permissionDecision"])
                    except (json.JSONDecodeError, TypeError):
                        pass  # stdout was not JSON -- normal for simple hooks

                elif r.returncode == 1:
                    # Block execution
                    result["blocked"] = True
                    reason = r.stderr.strip() or "Blocked by hook"
                    result["block_reason"] = reason
                    logging.info(f"  [hook:{event}] BLOCKED: {reason[:200]}")

                elif r.returncode == 2:
                    # Inject message
                    msg = r.stderr.strip()
                    if msg:
                        result["messages"].append(msg)
                        logging.info(f"  [hook:{event}] INJECT: {msg[:200]}")

            except subprocess.TimeoutExpired:
                logging.info(f"  [hook:{event}] Timeout ({HOOK_TIMEOUT}s)")
            except Exception as e:
                logging.error(f"  [hook:{event}] Error: {e}")

        return result


class SystemPromptBuilder:
    def __init__(self, workdir: Path = None, tools: list = None):
        self.workdir = workdir or WORKDIR
        self.tools = tools or []
        self.skills_dir = self.workdir / "skills"
        self.memory_dir = self.workdir / ".memory"

    def _build_core(self) -> str:
        return (
            f"You are a team lead at {self.workdir}. Spawn teammates and communicate via inboxes. "
            f"Manage teammates with shutdown and plan approval protocols.Teammates are autonomous -- they find work themselves.\n"
            # "Use background_run for long-running commands. "
            "Use task + worktree tools for multi-task work. "
            "For parallel or risky changes: create tasks, allocate worktree lanes, "
            "run commands in those lanes, then choose keep/remove for closeout."
            "\n\nYou can schedule future work with cron_create. "
            "Tasks fire automatically and their prompts are injected into the conversation.\n"
            "Use tools to solve tasks.\n"
            "You have both native tools and MCP tools available.\n"
            "MCP tools are prefixed with mcp__{server}__{tool}.\n"
            "All capabilities pass through the same permission gate before execution."

        )

    def _build_tool_listing(self) -> str:
        if not self.tools:
            return ""

        lines = ["# Available tools"]
        for tool in self.tools:
            props = tool.get("input_schema", {}).get("properties", {})
            params = ", ".join(props.keys())
            lines.append(f"- {tool['name']}({params}): {tool['description']}")
        return "\n".join(lines)

    def _build_skill_listing(self) -> str:
        # if not self.skills_dir.exists():
        #     return ""
        # skills = []
        # for skill_dir in sorted(self.skills_dir.iterdir()):
        #     skill_md = skill_dir / "SKILL.md"
        #     if not skill_md.exists():
        #         continue
        #     text = skill_md.read_text()
        #     match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
        #     if not match:
        #         continue
        #     meta = {}
        #     for line in match.group(1).splitlines():
        #         if ":" in line:
        #             k, _, v = line.partition(":")
        #             meta[k.strip()] = v.strip()
        #     name = meta.get("name", skill_md.stem)
        #     desc = meta.get("description", "")
        #     skills.append(f"- {name}: {desc}")
        # if not skills:
        #     return ""
        return "Use load_skill when a task needs specialized instructions before you act. \n# Available skills\n" + SKILL_REGISTRY.describe_available()

    def _build_memory_section(self) -> str:
        if not self.memory_dir.exists():
            return ""
        memories = []
        for md_file in sorted(self.memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue

            text = md_file.read_text()
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
            if not match:
                continue
            header, body = match.group(1), match.group(2)
            meta = {}
            for line in header.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
            name = meta.get("name", md_file.stem)
            mem_type = meta.get("type", "project")
            desc = meta.get("description", "")
            memories.append(f"[{mem_type}]: {name} {desc}\n{body}")
        if not memories:
            return ""
        memories.append(MEMORY_GUIDANCE)
        return "# Memories (persistent)\n\n" + "\n\n".join(memories)

    def _build_claude_md(self) -> str:
        """
        Load CLAUDE.md files in priority order (all are included):
        1. ~/.claude/CLAUDE.md (user-global instructions)
        2. <project-root>/CLAUDE.md (project instructions)
        3. <current-subdir>/CLAUDE.md (directory-specific instructions)
        """
        sources = []

        # User-global
        user_claude = Path.home() / ".claude" / "CLAUDE.md"
        if user_claude.exists():
            sources.append(("user global (~/.claude/CLAUDE.md)", user_claude.read_text()))

        # Project root
        project_claude = self.workdir / "CLAUDE.md"
        if project_claude.exists():
            sources.append(("project root (CLAUDE.md)", project_claude.read_text()))

        # Subdirectory -- in real CC, this walks from cwd up to project root
        # Teaching: check cwd if different from workdir
        cwd = Path.cwd()
        if cwd != self.workdir:
            subdir_claude = cwd / "CLAUDE.md"
            if subdir_claude.exists():
                sources.append((f"subdir ({cwd.name}/CLAUDE.md)", subdir_claude.read_text()))

        if not sources:
            return ""
        parts = ["# CLAUDE.md instructions"]
        for label, content in sources:
            parts.append(f"## From {label}")
            parts.append(content.strip())
        return "\n\n".join(parts)

    def _build_dynamic_context(self) -> str:
        lines = [
            f"Current date: {datetime.today().isoformat()}",
            f"Working directory: {self.workdir}",
            f"Model: {MODEL},"
            f"Platform: {platform.system()},"
        ]
        return "# Dynamic context\n" + "\n".join(lines)

    def build(self) -> str:
        """
        Assemble the full system prompt from all sections.

        Static sections (1-5) are separated from dynamic (6) by
        the DYNAMIC_BOUNDARY marker. In real CC, the static prefix
        is cached across turns to save prompt tokens.
        """
        sections = []

        core = self._build_core()
        if core:
            sections.append(core)

        tools = self._build_tool_listing()
        if tools:
            sections.append(tools)

        skills = self._build_skill_listing()
        if skills:
            sections.append(skills)

        memory = self._build_memory_section()
        if memory:
            sections.append(memory)

        claude_md = self._build_claude_md()
        if claude_md:
            sections.append(claude_md)

        # Static/dynamic boundary
        sections.append(DYNAMIC_BOUNDARY)

        dynamic = self._build_dynamic_context()
        if dynamic:
            sections.append(dynamic)

        return "\n\n".join(sections)


def build_system_reminder(extra: str = None) -> dict:
    parts = []
    if extra:
        parts.append(extra)
    if not parts:
        return None
    content = "<system-reminder>\n" + "\n".join(parts) + "\n</system-reminder>"
    return {"role": "user", "content": content}


class MemoryManager:
    """
    Load, build, and save persistent memories across sessions.

    The teaching version keeps memory explicit:
    one Markdown file per memory, plus one compact index file.
    """

    def __init__(self, memory_dir: Path = None):
        self.memory_dir = memory_dir or MEMORY_DIR
        self.memories = {}  # name -> {description, type, content}

    def load_all(self):
        """Load MEMORY.md index and all individual memory files."""
        self.memories = {}
        if not self.memory_dir.exists():
            return

        # Scan all .md files except MEMORY.md
        for md_file in sorted(self.memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue
            parsed = self._parse_frontmatter_memory(md_file.read_text())
            if parsed:
                name = parsed.get("name", md_file.stem)
                self.memories[name] = {
                    "description": parsed.get("description", ""),
                    "type": parsed.get("type", "project"),
                    "content": parsed.get("content", ""),
                    "file": md_file.name,
                }

        count = len(self.memories)
        if count > 0:
            logging.info(f"[Memory loaded: {count} memories from {self.memory_dir}]")

    # def load_memory_prompt(self) -> str:
    #     """Build a memory section for injection into the system prompt."""
    #     if not self.memories:
    #         return ""
    #
    #     sections = []
    #     sections.append("# Memories (persistent across sessions)")
    #     sections.append("")
    #
    #     # Group by type for readability
    #     for mem_type in MEMORY_TYPES:
    #         typed = {k: v for k, v in self.memories.items() if v["type"] == mem_type}
    #         if not typed:
    #             continue
    #         sections.append(f"## [{mem_type}]")
    #         for name, mem in typed.items():
    #             sections.append(f"### {name}: {mem['description']}")
    #             if mem["content"].strip():
    #                 sections.append(mem["content"].strip())
    #             sections.append("")
    #
    #     return "\n".join(sections)

    def save_memory(self, name: str, description: str, mem_type: str, content: str) -> str:
        """
        Save a memory to disk and update the index.

        Returns a status message.
        """
        if mem_type not in MEMORY_TYPES:
            return f"Error: type must be one of {MEMORY_TYPES}"

        # Sanitize name for filename
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower())
        if not safe_name:
            return "Error: invalid memory name"

        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Write individual memory file with frontmatter
        frontmatter = (
            f"---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"type: {mem_type}\n"
            f"---\n"
            f"{content}\n"
        )
        file_name = f"{safe_name}.md"
        file_path = self.memory_dir / file_name
        file_path.write_text(frontmatter)

        # Update in-memory store
        self.memories[name] = {
            "description": description,
            "type": mem_type,
            "content": content,
            "file": file_name,
        }

        # Rebuild MEMORY.md index
        self._rebuild_index()

        return f"Saved memory '{name}' [{mem_type}] to {file_path.relative_to(WORKDIR)}"

    def _rebuild_index(self):
        """Rebuild MEMORY.md from current in-memory state, capped at 200 lines."""
        lines = ["# Memory Index", ""]
        for name, mem in self.memories.items():
            lines.append(f"- {name}: {mem['description']} [{mem['type']}]")
            if len(lines) >= MAX_INDEX_LINES:
                lines.append(f"... (truncated at {MAX_INDEX_LINES} lines)")
                break
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        MEMORY_INDEX.write_text("\n".join(lines) + "\n")

    def _parse_frontmatter_memory(self, text: str) -> dict | None:
        """Parse --- delimited frontmatter + body content."""
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if not match:
            return None
        header, body = match.group(1), match.group(2)
        result = {"content": body.strip()}
        for line in header.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip()
        return result


def estimate_tokens(messages: list) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(json.dumps(messages, default=str)) // 4


def auto_compact(messages: list) -> list:
    conversation_text = json.dumps(messages, default=str)
    prompt = (
            "Summarize this conversation for continuity. Include:\n"
            "1) Task overview and success criteria\n"
            "2) Current state: completed work, files touched\n"
            "3) Key decisions and failed approaches\n"
            "4) Remaining next steps\n"
            "Be concise but preserve critical details.\n\n"
            + conversation_text
    )
    try:
        response = client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        summary = response.content[0]
    except Exception as e:
        summary = f"(compact failed: {e}). Previous context lost."

    continuation = (
        "This session continues from a previous conversation that was compacted. "
        f"Summary of prior context:\n\n{summary}\n\n"
        "Continue from where we left off without re-asking the user."
    )
    return [{"role": "user", "content": continuation}]


def backoff_delay(attempt: int) -> float:
    delay = min(BACKOFF_BASE_DELAY * (2 ** attempt), BACKOFF_MAX_DELAY)
    jitter = delay * random.uniform(0, 0.1)
    return delay + jitter


# -- EventBus: append-only lifecycle events for observability --
class EventBus:
    def __init__(self, event_log_path: Path) -> None:
        self.path = event_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def emit(self, event: str, task_id=None, wt_name=None, error=None, **extra) -> None:
        payload = {"event": event, "ts": time.time()}
        if task_id is not None:
            payload["task_id"] = task_id
        if wt_name:
            payload["wt_name"] = wt_name
        if error:
            payload["error"] = error
        payload.update(extra)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def list_recent(self, limit: int = 20) -> str:
        n = max(1, min(int(limit or 20), 200))
        items = []
        for line in self.path.read_text().splitlines():
            try:
                items.append(json.loads(line))
            except Exception as e:
                logging.error(f"error: {e}")
                items.append({"event": "parse_error", "raw": line})
        return json.dumps(items, ensure_ascii=False, indent=2)


# -- TaskManager: CRUD for a persistent task graph --
class TaskManager:
    """Persistent TaskRecord store.

    Think "work graph on disk", not "currently running worker".
    """

    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _path(self, task_id: int):
        return self.dir / f"task_{task_id}.json"

    def _load(self, task_id: int) -> dict:
        path = self._path(task_id)
        if not path.exists():
            raise FileNotFoundError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        path = self._path(task["id"])
        path.write_text(json.dumps(task, default=str, indent=2, ensure_ascii=False), encoding="utf-8")

    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "blockedBy": [],
            "blocks": [],
            "owner": "",
            "worktree": "",
            "worktree_state": "unbound",
            "last_worktree": "",
            "closeout": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2)

    def exists(self, task_id: int) -> bool:
        return self._path(task_id).exists()

    def update(self, task_id: int, status: str = None, owner: str = None,
               add_blocked_by: list = None, add_blocks: list = None) -> str:
        task = self._load(task_id)
        if owner is not None:
            task["owner"] = owner
        if status:
            if status not in ("pending", "in_progress", "completed", "deleted"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            if status == "completed":
                self._clear_dependency(task_id)
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))
            for blocked_id in add_blocks:
                try:
                    blocked = self._load(blocked_id)
                    if task_id not in blocked["blockedBy"]:
                        blocked["blockedBy"].append(task_id)
                        self._save(blocked)
                except Exception as e:
                    logging.error(f"blocked error: {e}")
                    pass
        self._save(task)
        return json.dumps(task, indent=2)

    def _clear_dependency(self, completed_id: int):
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task["blockedBy"]:
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> str:
        task = self._load(task_id)
        task["worktree"] = worktree
        task["last_worktree"] = worktree
        task["worktree_state"] = "active"
        if owner:
            task["owner"] = owner
        if task["status"] == "pending":
            task["status"] = "is_progress"
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def record_closeout(self, task_id: int, action: str, reason: str = "", keep_binding: bool = False) -> str:
        task = self._load(task_id)
        task["closeout"] = {
            "action": action,
            "reason": reason,
            "at": time.time(),
        }
        task["worktree_state"] = action
        task["updated_at"] = time.time()
        if not keep_binding:
            task["worktree"] = ""
        self._save(task)
        return json.dumps(task, indent=2)

    def list_all(self):
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            return "No tasks."

        lines = []
        for t in tasks:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
                "deleted": "[-]"
            }.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            owner = f" owner={t['owner']}" if t.get("owner") else ""
            wt = f" worktree={t['worktree']}" if t.get("worktree") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{owner}{blocked}{wt}")
        return "\n".join(lines)


TASKS = TaskManager(TASKS_DIR)
EVENTS = EventBus(REPO_ROOT / ".worktrees" / "events.jsonl")


# -- WorktreeManager: create/list/run/remove git worktrees --
class WorktreeManager:
    def __init__(self, repo_root: Path, tasks: TaskManager, events: EventBus) -> None:
        self.repo_root = repo_root
        self.tasks = tasks
        self.events = events
        self.dir = repo_root / "worktrees"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"worktrees": []}, indent=2))
        self.git_available = self._check_git()

    def _check_git(self) -> bool:
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_root, capture_output=True, text=True, timeout=10,
            )
            return r.returncode == 0
        except Exception:
            return False

    def _run_git(self, args: list[str]) -> str:
        if not self.git_available:
            raise RuntimeError("Not in a git repository.")
        r = subprocess.run(
            ["git", *args], cwd=self.repo_root,
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode != 0:
            raise RuntimeError((r.stdout + r.stderr).strip() or f"git {' '.join(args)} failed")
        return (r.stdout + r.stderr).strip() or "(no output)"

    def _load_index(self) -> dict:
        return json.loads(self.index_path.read_text())

    def _save_index(self, data: dict):
        self.index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _find(self, name: str) -> dict | None:
        for wt in self._load_index().get("worktrees", []):
            if wt["name"] == name:
                return wt
        return None

    def _update_entry(self, name: str, **changes) -> dict:
        idx = self._load_index()
        updated = None
        for item in idx.get("worktrees", []):
            if item["name"] == name:
                item.update(changes)
                updated = item
                break
        self._save_index(idx)
        if not updated:
            raise ValueError(f"Worktree '{name}' not found in index")
        return updated

    def _validate_name(self, name: str):
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", name or ""):
            raise ValueError("Invalid worktree name. Use 1-40 chars: letters, digits, ., _, -")

    def create(self, name: str, task_id: int = None, base_ref: str = "HEAD") -> str:
        self._validate_name(name)
        if self._find(name):
            raise ValueError(f"Worktree '{name}' already exists")
        if task_id is not None and not self.tasks.exists(task_id):
            raise ValueError(f"Task {task_id} not found")

        path = self.dir / name
        branch = f"wt/{name}"
        self.events.emit("worktree.create.before", task_id=task_id, wt_name=name)
        try:
            self._run_git(["worktree", "add", "-b", branch, str(path), base_ref])
            entry = {
                "name": name,
                "path": str(path),
                "branch": branch,
                "task_id": task_id,
                "status": "active",
                "create_at": time.time(),
            }
            idx = self._load_index()
            idx["worktrees"].append(entry)
            self._save_index(idx)
            if task_id is not None:
                self.tasks.bind_worktree(task_id, name)
            self.events.emit("worktree.create.after", task_id=task_id, wt_name=name)
            return json.dumps(entry, indent=2)
        except Exception as e:
            self.events.emit("worktree.create.failed", task_id=task_id, wt_name=name, error=str(e))
            raise

    def list_all(self) -> str:
        wts = self._load_index().get("worktrees", [])
        if not wts:
            return "No worktrees in index."
        lines = []
        for wt in wts:
            suffix = f" task={wt['task_id']}" if wt.get("task_id") else ""
            lines.append(f"[{wt.get('status', '?')}] {wt['name']} -> {wt['path']} ({wt.get('branch', '-')}){suffix}")
        return "\n".join(lines)

    def status(self, name: str) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
        r = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=path, capture_output=True, text=True, timeout=60,
        )
        return (r.stdout + r.stderr).strip() or "Clean worktree"

    def enter(self, name: str) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
        updated = self._update_entry(name, last_entered_at=time.time())
        self.events.emit("worktree.enter", task_id=wt.get("task_id", None), wt_name=name, path=str(path))
        return json.dumps(updated, indent=2)

    def run(self, name: str, command: str) -> str:
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
        try:
            self._update_entry(
                name,
                last_entered_at=time.time(),
                last_command_at=time.time(),
                last_command_preview=command[:120],
            )
            self.events.emit("worktree.run.before", task_id=wt.get("task_id", None),
                             wt_name=name, command=command[:120])
            r = subprocess.run(command, shell=True, cwd=path, capture_output=True, text=True, timeout=300)
            out = (r.stdout + r.stderr).strip()
            self.events.emit("worktree.run.after", task_id=wt.get("task_id", None), wt_name=name)
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:

            self.events.emit("worktree.run.timeout", task_id=wt.get("task_id"), wt_name=name)
            return "Error: Timeout (300s)"
        except Exception as e:
            logging.error(f"worktree.run error: {e}")
            return "worktree.run error"

    def remove(self, name: str, force: bool = False,
               complete_task: bool = False, reason: str = ""
               ) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        task_id = wt.get("task_id")
        self.events.emit("worktree.remove.before", task_id=task_id, wt_name=name)
        try:
            args = ["worktree", "remove"]
            if force:
                args.append("--force")
            args.append(wt["path"])
            self._run_git(args)
            if complete_task and task_id is not None:
                self.tasks.update(task_id, status="completed")
                self.events.emit("task.completed", task_id=task_id, wt_name=name)
            if task_id is not None:
                self.tasks.record_closeout(task_id, "removed", reason, keep_binding=False)
            self._update_entry(
                name,
                status="removed",
                removed_at=time.time(),
                closeout={"action": "remove", "reason": reason, "at": time.time()},
            )
            self.events.emit("worktree.remove.after", task_id=task_id, wt_name=name)
            return f"Removed worktree '{name}'"
        except Exception as e:
            logging.error(f"worktree.remove error: {e}")
            self.events.emit("worktree.remove.failed", task_id=task_id, wt_name=name, error=str(e))
            raise

    def keep(self, name: str) -> str:
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        if wt.get("task_id") is not None:
            self.tasks.record_closeout(wt["task_id"], "kept", "", keep_binding=True)
        self._update_entry(
            name,
            status="kept",
            kept_at=time.time(),
            closeout={"action": "kept", "reason": "", "at": time.time()},
        )
        self.events.emit("worktree.keep", task_id=wt.get("task_id"), wt_name=name)
        return json.dumps(self._find(name), indent=2)

    def closeout(
            self,
            name: str,
            action: str,
            reason: str = "",
            force: bool = False,
            complete_task: bool = False,
    ) -> str:
        if action == "keep":
            wt = self._find(name)
            if not wt:
                return f"Error: Unknown worktree '{name}'"
            if wt.get("task_id") is not None:
                self.tasks.record_closeout(
                    wt["task_id"], "kept", reason, keep_binding=True
                )
                if complete_task:
                    self.tasks.update(wt["task_id"], status="completed")
            self._update_entry(
                name,
                status="kept",
                kept_at=time.time(),
                closeout={"action": "keep", "reason": reason, "at": time.time()},
            )
            self.events.emit(
                "worktree.closeout.keep",
                task_id=wt.get("task_id"),
                wt_name=name,
                reason=reason,
            )
            return json.dumps(self._find(name), indent=2)
        if action == "remove":
            self.events.emit("worktree.closeout.remove", wt_name=name, reason=reason)
            return self.remove(
                name,
                force=force,
                complete_task=complete_task,
                reason=reason,
            )
        raise ValueError("action must be 'keep' or 'remove'")


WORKTREES = WorktreeManager(REPO_ROOT, TASKS, EVENTS)


class NotificationQueue:
    """
    Priority-based notification queue with same-key folding.

    Folding means a newer message can replace an older message with the
    same key, so the context is not flooded with stale updates.
    """

    PRIORITIES = {"immediate": 0, "high": 1, "medium": 2, "low": 3}

    def __init__(self):
        self._queue = []  # list of (priority, key, message)
        self._lock = threading.Lock()

    def push(self, message: str, priority: str = "medium", key: str = None):
        """
        Add a message to the queue, folding if key matches an existing entry.
        :param message:
        :param priority:
        :param key:
        :return:
        """

        with self._lock:
            if key:
                # Fold: replace existing message with same key
                self._queue = [(p, k, m) for p, k, m in self._queue if k != key]
            self._queue.append((self.PRIORITIES.get(priority, 2), key, message))
            self._queue.sort(key=lambda x: x[0])

    def drain(self) -> list[str]:
        """
        Return all pending messages in priority order and clear the queue.
        :return:
        """
        with self._lock:
            messages = [m for _, _, m in self._queue]
            self._queue.clear()
            return messages


# -- BackgroundManager: threaded execution + notification queue --
class BackroundManager:
    def __init__(self):
        self.dir = RUNTIME_DIR
        self.tasks = {}
        self._notification_queue = []
        self._lock = threading.Lock()

    def _record_path(self, task_id: str) -> Path:
        return self.dir / f"{task_id}.json"

    def _output_path(self, task_id: str) -> Path:
        return self.dir / f"{task_id}.log"

    def _persist_task(self, task_id: str) -> None:
        record = self.tasks.get(task_id, {})
        self._record_path(task_id).write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _preview(self, output: str, limit: int = 500) -> str:
        compact = " ".join((output or "(no output)").split())
        return compact[:limit]

    def run(self, command: str) -> str:
        """Start a background thread, return task_id immediately."""
        task_id = str(uuid.uuid4())[:8]
        output_file = self._output_path(task_id)
        self.tasks[task_id] = {
            "id": task_id,
            "status": "running",
            "result": None,
            "command": command,
            "started_at": time.time(),
            "finished_at": None,
            "result_preview": "",
            "output_file": str(output_file.relative_to(WORKDIR)),
        }
        self._persist_task(task_id)
        thread = threading.Thread(
            target=self._execute, args=(task_id, command), daemon=True
        )
        thread.start()
        return (
            f"Background task {task_id} started: {command[:80]} "
            f"(output_file={output_file.relative_to(WORKDIR)})"
        )

    def _execute(self, task_id: str, command: str):
        """Thread target: run subprocess, capture output, push to queue."""
        try:
            r = subprocess.run(
                command, shell=True, cwd=WORKDIR,
                capture_output=True, text=True, timeout=300
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            output = f"Error: {e}"
            status = "error"
        final_output = output or "(no output)"
        preview = self._preview(final_output)
        output_path = self._output_path(task_id)
        output_path.write_text(final_output)
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = final_output
        self.tasks[task_id]["finished_at"] = time.time()
        self.tasks[task_id]["result_preview"] = preview
        self._persist_task(task_id)
        with self._lock:
            self._notification_queue.append({
                "task_id": task_id,
                "status": status,
                "command": command[:80],
                "preview": preview,
                "output_file": str(output_path.relative_to(WORKDIR)),
            })

    def check(self, task_id: str = None) -> str:
        """Check status of one task or list all."""
        if task_id:
            t = self.tasks.get(task_id)
            if not t:
                return f"Error: Unknown task {task_id}"
            visible = {
                "id": t["id"],
                "status": t["status"],
                "command": t["command"],
                "result_preview": t.get("result_preview", ""),
                "output_file": t.get("output_file", ""),
            }
            return json.dumps(visible, indent=2, ensure_ascii=False)
        lines = []
        for tid, t in self.tasks.items():
            lines.append(
                f"{tid}: [{t['status']}] {t['command'][:60]} "
                f"-> {t.get('result_preview') or '(running)'}"
            )
            return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list:
        """Return and clear all pending completion notifications."""
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs

    def detect_stalled(self) -> list[str]:
        """
        Return task IDs that have been running longer than STALL_THRESHOLD_S.
        :return:
        """
        now = time.time()
        stalled = []
        for task_id, info in self.tasks.items():
            if info["status"] != "running":
                continue
            elapsed = now - info.get("started_at", now)
            if elapsed > STALL_THRESHOLD_S:
                stalled.append(task_id)
        return stalled


BG = BackroundManager()


class CronLock:
    """
    PID-file-based lock to prevent multiple sessions from firing the same cron job.
    """

    def __init__(self, lock_path: Path = None):
        self._lock_path = lock_path or CRON_LOCK_FILE

    def acquire(self) -> bool:
        """
        Try to acquire the cron lock. Returns True on success.

        If a lock file exists, check whether the PID inside is still alive.
        If the process is dead the lock is stale and we can take over.
        """
        if self._lock_path.exists():
            try:
                stored_pid = int(self._lock_path.read_text().strip())
                # PID liveness probe: send signal 0 (no-op) to check existence
                os.kill(stored_pid, 0)
                # Process is alive -- lock is held by another session
                return False
            except (ValueError, ProcessLookupError, PermissionError, OSError):
                # Stale lock (process dead or PID unparseable) -- remove it
                pass
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.write_text(str(os.getpid()))
        return True

    def release(self):
        """Remove the lock file if it belongs to this process."""
        try:
            if self._lock_path.exists():
                stored_pid = int(self._lock_path.read_text().strip())
                if stored_pid == os.getpid():
                    self._lock_path.unlink()
        except (ValueError, OSError):
            pass


def cron_matches(expr: str, dt: datetime) -> bool:
    """
    Check if a 5-field cron expression matches a given datetime.

    Fields: minute hour day-of-month month day-of-week
    Supports: * (any), */N (every N), N (exact), N-M (range), N,M (list)

    No external dependencies -- simple manual matching.
    """
    fields = expr.strip().split()
    if len(fields) != 5:
        return False

    values = [dt.minute, dt.hour, dt.day, dt.month, dt.weekday()]
    # Python weekday: 0=Monday; cron: 0=Sunday. Convert.
    cron_dow = (dt.weekday() + 1) % 7
    values[4] = cron_dow
    ranges = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]

    for field, value, (lo, hi) in zip(fields, values, ranges):
        if not _field_matches(field, value, lo, hi):
            return False
    return True


def _field_matches(field: str, value: int, lo: int, hi: int) -> bool:
    """Match a single cron field against a value."""
    if field == "*":
        return True

    for part in field.split(","):
        # Handle step: */N or N-M/S
        step = 1
        if "/" in part:
            part, step_str = part.split("/", 1)
            step = int(step_str)

        if part == "*":
            # */N -- check if value is on the step grid
            if (value - lo) % step == 0:
                return True
        elif "-" in part:
            # Range: N-M
            start, end = part.split("-", 1)
            start, end = int(start), int(end)
            if start <= value <= end and (value - start) % step == 0:
                return True
        else:
            # Exact value
            if int(part) == value:
                return True

    return False


class CronScheduler:
    """
    Manage scheduled tasks with background checking.

    Teaching version keeps only the core pieces: schedule records, a
    minute checker, optional persistence, and a notification queue.
    """

    def __init__(self):
        self.tasks = []  # list of task dicts
        self.queue = Queue()  # notification queue
        self._stop_event = threading.Event()
        self._thread = None
        self._last_check_minute = -1  # avoid double-firing within same minute

    def start(self) -> None:
        """Load durable tasks and start the background check thread."""
        self._load_durable()
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()
        count = len(self.tasks)
        if count:
            logging.info(f"[Cron] Loaded {count} scheduled tasks")

    def stop(self):
        """Stop the background thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def create(self, cron_expr: str, prompt: str,
               recurring: bool = True, durable: bool = False) -> str:
        task_id = str(uuid.uuid4())[:8]
        now = time.time()
        task = {
            "id": task_id,
            "cron": cron_expr,
            "prompt": prompt,
            "recurring": recurring,
            "durable": durable,
            "createdAt": now,
        }

        # Jitter for recurring tasks: if the cron fires on :00 or :30,
        # note it so we can offset the check slightly
        if recurring:
            task["jitter_offset"] = self._compute_jitter(cron_expr)

        self.tasks.append(task)
        if durable:
            self._save_durable()
        mode = "recurring" if recurring else "one-shot"
        store = "durable" if durable else "session-onle"
        return f"Created task {task_id} ({mode}, {store}): cron={cron_expr}"

    def delete(self, task_id: str) -> str:
        """Delete a scheduled task by ID."""
        before = len(self.tasks)
        self.tasks = [t for t in self.tasks if t["id"] != task_id]
        if len(self.tasks) < before:
            self._save_durable()
            return f"Deleted task {task_id}"
        return f"Task {task_id} not found"

    def list_tasks(self) -> str:
        """List all scheduled tasks."""
        if not self.tasks:
            return "No scheduled tasks."
        lines = []
        for t in self.tasks:
            mode = "recurring" if t["recurring"] else "one-shot"
            store = "durable" if t["durable"] else "session"
            age_hours = (time.time() - t["createdAt"]) / 3600
            lines.append(
                f"  {t['id']}  {t['cron']}  [{mode}/{store}] "
                f"({age_hours:.1f}h old): {t['prompt'][:60]}"
            )
        return "\n".join(lines)

    def drain_notifications(self) -> list[str]:
        """Drain all pending notifications from the queue."""
        notifications = []
        while True:
            try:
                notifications.append(self.queue.get_nowait())
            except Empty:
                break
        return notifications

    def _save_durable(self):
        """Save durable tasks to disk."""
        durable = [t for t in self.tasks if t.get("durable")]
        SCHEDULED_TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SCHEDULED_TASKS_FILE.write_text(
            json.dumps(durable, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _compute_jitter(self, cron_expr: str) -> int:
        """If cron targets :00 or :30, return a small offset (1-4 minutes)."""
        fields = cron_expr.strip().split()
        if len(fields) < 1:
            return 0
        minute_field = fields[0]
        try:
            minute_val = int(minute_field)
            if minute_val in JITTER_MINUTES:
                # Deterministic jitter based on the expression hash
                return (hash(cron_expr) % JITTER_OFFSET_MAX) + 1
        except ValueError:
            pass
        return 0

    def _load_durable(self):
        """Load durable tasks from .claude/scheduled_tasks.json."""
        if not SCHEDULED_TASKS_FILE.exists():
            return
        try:
            data = json.loads(SCHEDULED_TASKS_FILE.read_text())
            self.tasks = [t for t in data if t.get("durable")]
        except Exception as e:
            logging.error(f"[Cron] Error loading tasks: {e}")

    def _check_loop(self):
        """Background thread: check every second if any task is due."""
        while not self._stop_event.is_set():
            now = datetime.now()
            current_minute = now.hour * 60 + now.minute

            # Only check once per minute to avoid double-firing
            if current_minute != self._last_check_minute:
                self._last_check_minute = current_minute
                self._check_tasks(now)

            self._stop_event.wait(timeout=1)

    def _check_tasks(self, now: datetime):
        """Check all tasks against current time, fire matches."""
        expired = []
        fired_oneshots = []

        for task in self.tasks:
            # Auto-expiry: recurring tasks older than 7 days
            age_days = (time.time() - task["createdAt"]) / 86400
            if task["recurring"] and age_days > AUTO_EXPIRY_DAYS:
                expired.append(task["id"])
                continue

            # Apply jitter offset for the match check
            check_time = now
            jitter = task.get("jitter_offset", 0)
            if jitter:
                check_time = now - timedelta(minutes=jitter)

            if cron_matches(task["cron"], check_time):
                notification = (
                    f"[Scheduled task {task['id']}]: {task['prompt']}"
                )
                self.queue.put(notification)
                task["last_fired"] = time.time()
                # logging.info(f"[Cron] Fired: {task['id']}")

                if not task["recurring"]:
                    fired_oneshots.append(task["id"])

        # Clean up expired and one-shot tasks
        if expired or fired_oneshots:
            remove_ids = set(expired) | set(fired_oneshots)
            self.tasks = [t for t in self.tasks if t["id"] not in remove_ids]
            for tid in expired:
                logging.info(f"[Cron] Auto-expired: {tid} (older than {AUTO_EXPIRY_DAYS} days)")
            for tid in fired_oneshots:
                logging.info(f"[Cron] One-shot completed and removed: {tid}")
            self._save_durable()

    def detect_missed_tasks(self) -> list[dict]:
        """
        On startup, check each durable task's last_fired time.

        If a task should have fired while the session was closed (i.e.
        the gap between last_fired and now contains at least one cron match),
        flag it as missed. The caller can then let the user decide whether
        to run or discard each missed task.

        """
        now = datetime.now()
        missed = []
        for task in self.tasks:
            last_fired = task.get("last_fired")
            if last_fired is None:
                continue
            last_dt = datetime.fromtimestamp(last_fired)
            # Walk forward minute-by-minute from last_fired to now (cap at 24h)
            check = last_dt + timedelta(minutes=1)
            cap = min(now, last_dt + timedelta(hours=24))
            while check <= cap:
                if cron_matches(task["cron"], check):
                    missed.append({
                        "id": task["id"],
                        "cron": task["cron"],
                        "prompt": task["prompt"],
                        "missed_at": check.isoformat(),
                    })
                    break  # one miss is enough to flag it
                check += timedelta(minutes=1)
        return missed


scheduler = CronScheduler()


# -- MessageBus: JSONL inbox per teammate --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                messages.append(json.loads(line.strip()))
        inbox_path.write_text("")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, msg_type="broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


class RequestStore:
    """
    Durable request records for protocol workflows.

    Protocol state should survive long enough to inspect, resume, or reconcile.
    This store keeps one JSON file per request_id under .team/requests/.
    """

    def __init__(self, base_dir: Path):
        self.dir = base_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, request_id: str) -> Path:
        return self.dir / f"{request_id}.json"

    def create(self, record: dict) -> dict:
        request_id = record["request_id"]
        with self._lock:
            self._path(request_id).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    def get(self, request_id: str) -> dict | None:
        path = self._path(request_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def update(self, request_id: str, **changes) -> dict | None:
        record = self.get(request_id)
        if not record:
            return None
        record.update(changes)
        record["updated_at"] = time.time()
        with self._lock:
            self._path(request_id).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record


REQUEST_STORE = RequestStore(REQUESTS_DIR)


# --- Task board scanning --
def _append_clain_event(payload: dict):
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    with CLAIM_EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _task_allows_role(task: dict, role: str | None) -> bool:
    required_role = task.get("claim_role") or task.get("required_role") or ""
    if not required_role:
        return True
    return bool(role) and role == required_role


def is_claimable_task(task: dict, role: str | None = None) -> bool:
    if task.get("status") == "pending" and not task.get("owner") and not task.get("blockedBy") and _task_allows_role(
            task, role):
        return True
    else:
        return False


def scan_unclaimed_tasks(role: str | None = None) -> list:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    unclaimed_tasks = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text(encoding="utf-8"))
        if is_claimable_task(task, role):
            unclaimed_tasks.append(task)
    return unclaimed_tasks


def claim_task(
        task_id: int,
        owner: str,
        role: str | None = None,
        source: str = "manual",
) -> str:
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text(encoding="utf-8"))
        if not is_claimable_task(task, role):
            return f"Error: Task {task_id} is not claimable for role={role or '(any)'}"
        task["owner"] = owner
        task["status"] = "in_progress"
        task["claim_at"] = time.time()
        task["claim_source"] = source
        path.write_text(json.dumps(task, indent=2), encoding="utf-8")
    _append_clain_event({
        "event": "task.claimed",
        "task_id": task_id,
        "owner": owner,
        "role": role,
        "source": source,
        "ts": time.time()
    })
    return f"Claimed task #{task_id} for {owner} via {source}"


def make_identity_block(name: str, role: str, team_name: str) -> dict:
    return {
        "role": "user",
        "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>",
    }


def ensure_identity_content(messages: list, name: str, role: str, team_name: str):
    if messages and "<identity>" in str(messages[0].get("content", "")):
        return
    messages.insert(0, make_identity_block(name, role, team_name))
    messages.insert(1, {"role": "assistant", "content": f"I am {name}, Continuing."})


# -- TeammateManager: persistent named agents with config.json --
class TeammateManager:
    def __init__(self, team_dir):
        self.dir = team_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self):
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8")

    def _find_member(self, name):
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def _set_status(self, name: str, status: str):
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned {name} (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        team_name = self.config["team_name"]
        sys_prompt = (
            f"You are '{name}', role: {role}, team: {team_name}， at {WORKDIR}. "
            f"Use idle tool when you have no more work. You will auto-claim new tasks."
            f"Use send_message to communicate. Complete your task."
            f"Submit plans via plan_approval before major work. "
            f"Respond to shutdown_request with shutdown_response."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()
        while True:
            # -- WORK PHASH: standard agent loop
            for _ in range(60):
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
                    messages.append({"role": "user", "content": json.dumps(msg)})
                try:
                    response = client.messages.create(
                        model=MODEL,
                        system=sys_prompt,
                        messages=messages,
                        tools=tools,
                        max_tokens=8000
                    )
                except Exception as e:
                    self._set_status(name, "idle")
                    logging.error(f"teammate {name} -Error: {e}; messages: {messages}")
                    return
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    logging.info(f"teamagent stop_reason:  {response.stop_reason}")
                    break
                results = []
                idle_requested = False
                for block in response.content:
                    if block.type == "tool_use":
                        if block.name == "idle":
                            idle_requested = True
                            output = "Entering idle phase. Will poll for new tasks."
                        else:
                            output = self._exec(name, block.name, block.input)
                        logging.info(f"  teammate tool--  [{name}] {block.name}: {str(output)[:200]}")
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(output)
                        })
                messages.append({"role": "user", "content": results})
                if idle_requested:
                    break

            # -- IDLE PHASE: poll for inbox messages and unclaimed tasks --
            self._set_status(name, "idle")
            resume = False
            polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1)
            for _ in range(polls):
                time.sleep(POLL_INTERVAL)
                inbox = BUS.read_inbox(name)
                if inbox:
                    ensure_identity_content(messages, name, role, team_name)
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break
                unclaimed = scan_unclaimed_tasks(role)
                if unclaimed:
                    task = unclaimed[0]
                    claim_result = claim_task(task_id=task["id"], owner=name, role=role, source="auto")
                    if claim_result.startswith("Error"):
                        continue
                    task_prompt = (
                        f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
                        f"{task.get('description', '')}</auto-claimed>"
                    )
                    ensure_identity_content(messages, name, role, team_name)
                    messages.append({"role": "user", "content": task_prompt})
                    messages.append({"role": "assistant", "content": f"{claim_result}. Working on it."})
                    resume = True
                    break

            if not resume:
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        args.update({"sender": sender})
        handler = self._teammate_tool_handlers().get(tool_name)
        try:
            output = handler(**args) if handler else f"Unknown tool: {tool_name}"
        except Exception as e:
            output = f"Tool _exec Error: {e}"
        return output

    def _teammate_tools(self) -> list:
        # these base tools are unchanged from s02
        return [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object", "properties": {"command": {"type": "string"}},
                              "required": ["command"]}},
            {"name": "read_file", "description": "Read file contents.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "write_file", "description": "Write content to file.",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                              "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Replace exact text in file.",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"}, "old_text": {"type": "string"},
                                             "new_text": {"type": "string"}},
                              "required": ["path", "old_text", "new_text"]}},
            {"name": "send_message", "description": "Send message to a teammate.",
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"},
                                                               "msg_type": {"type": "string",
                                                                            "enum": list(VALID_MSG_TYPES)}},
                              "required": ["to", "content"]}},
            {"name": "read_inbox", "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
            {"name": "shutdown_response",
             "description": "Respond to a shutdown request. Approve to shut down, reject to keep working.",
             "input_schema": {"type": "object",
                              "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"},
                                             "reason": {"type": "string"}}, "required": ["request_id", "approve"]}},
            {"name": "plan_approval", "description": "Submit a plan for lead approval. Provide plan text.",
             "input_schema": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}},
            {"name": "idle", "description": "Signal that you have no more work. Enters idle polling phase.",
             "input_schema": {"type": "object", "properties": {}}},
            {"name": "claim_task", "description": "Claim a task from the task board by ID.",
             "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}},
                              "required": ["task_id"]}},
        ]

    def _teammate_tool_handlers(self) -> dict:
        return {
            "bash": lambda **kw: run_bash(kw["command"]),
            "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
            "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
            "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
            "send_message": lambda **kw: BUS.send(kw["sender"], kw["to"], kw["content"], kw.get("msg_type", "message")),
            "read_inbox": lambda **kw: json.dumps(BUS.read_inbox(kw["sender"]), indent=2),
            "shutdown_response": lambda **kw: self._shutdown_response(kw["request_id"], kw["approve"], kw["sender"],
                                                                      kw.get("reason", "")),
            "plan_approval": lambda **kw: self._plan_approval(kw["plan"], kw.get("sender", "")),
            "claim_task": lambda **kw: claim_task(kw["task_id"], kw["sender"], kw["to"],
                                                  role=self._find_member(kw["sender"]).get("role") if self._find_member(
                                                      kw["sender"]) else None, source="manual"),
        }

    def _shutdown_response(self, request_id: str, approve: bool, sender: str, reason: str = ""):
        updated = REQUEST_STORE.update(
            request_id,
            status="approved" if approve else "rejected",
            resolved_by=sender,
            resolved_at=time.time(),
            response={"approve": approve, "reason": reason},
        )
        if not updated:
            return f"Error: Unknown shutdown request {request_id}"
        BUS.send(
            sender, "lead", reason,
            "shutdown_response", {"request_id": request_id, "approve": approve},
        )
        return f"Shutdown {'approved' if approve else 'rejected'}"

    def _plan_approval(self, plan: str, sender: str):
        req_id = str(uuid.uuid4())[:8]
        REQUEST_STORE.create({
            "request_id": req_id,
            "kind": "plan_approval",
            "from": sender,
            "to": "lead",
            "status": "pending",
            "plan": plan,
            "created_at": time.time(),
            "updated_at": time.time(),
        })
        BUS.send(
            sender, "lead", plan, "plan_approval",
            {"request_id": req_id, "plan": plan},
        )
        return f"Plan submitted (request_id={req_id}). Waiting for lead approval."

    def list_all(self):
        if not self.config["members"]:
            return "No teammates"
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(TEAM_DIR)


class MCPClient:
    """
    Minimal MCP client over stdio.

    This is enough to teach the core architecture without dragging readers
    through every transport, auth flow, or marketplace detail up front.
    """

    def __init__(self, server_name: str, command: str, args: list = None, env: dict = None):
        self.server_name = server_name
        self.command = command
        self.args = args or []
        self.env = {**os.environ, **(env or {})}
        self.process = None
        self._request_id = 0
        self._tools = []

    def connect(self):
        """Start the MCP server process."""
        try:
            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                text=True,
            )
            # Send initialize request
            self._send({"method": "initialize", "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "teaching-agent", "version": "1.0"},
            }})
            response = self._recv()
            if response and "result" in response:
                # Send initialized notification
                self._send({"method": "notifications/initialized"})
                return True
        except FileNotFoundError:
            logging.error(f"[MCP] Server command not found: {self.command}")
        except Exception as e:
            logging.error(f"[MCP] Connection failed: {e}")
        return False

    def list_tools(self) -> list:
        """Fetch available tools from the server."""
        self._send({"method": "list/tools", "params": {}})
        response = self._recv()
        if response and "result" in response:
            self._tools = response["result"].get("tools", [])
        return self._tools

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool on the server."""
        self._send({"method": "tool/call", "params": {
            "name": tool_name,
            "arguments": arguments,
        }})
        response = self._recv()
        if response and "result" in response:
            content = response["result"].get("content", [])
            return "\n".join(c.get("text", str(c)) for c in content)
        if response and "error" in response:
            return f"MCP Error: {response['error'].get('message', 'unknown')}"
        return "MCP Error: no response"

    def get_agent_tools(self) -> list:
        """
        Convert MCP tools to agent tool format.

        Teaching version uses the same simple prefix idea:
        mcp__{server_name}__{tool_name}
        :return:
        """
        agent_tools = []
        for tool in self._tools:
            prefixed_name = f"mcp__{self.server_name}__{tool['name']}"
            agent_tools.append({
                "name": prefixed_name,
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema", {"type": "object", "properties": {}}),
                "_mcp_server": self.server_name,
                "_mcp_tool": tool["name"],
            })
            return agent_tools

    def disconnect(self):
        """Shut down the server process."""
        if self.process:
            try:
                self._send({"method": "shutdown"})
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None

    def _send(self, message: dict):
        if not self.process or self.process.poll() is not None:
            return
        self._request_id += 1
        envelope = {"jsonrpc": "2.0", "id": self._request_id, **message}
        line = json.dumps(envelope) + "\n"
        try:
            self.process.stdin.write(line)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _recv(self) -> dict | None:
        if not self.process or self.process.poll() is not None:
            return None
        try:
            line = self.process.stdout.readline()
            if line:
                return json.loads(line)
        except (json.JSONDecodeError, OSError):
            pass
        return None


class PluginLoader:
    """
    Load plugins from .claude-plugin/ directories.

    Teaching version implements the smallest useful plugin flow:
    read a manifest, discover MCP server configs, and register them.
    """

    def __init__(self, search_dirs: list = None):
        self.search_dirs = search_dirs or [WORKDIR]
        self.plugins = {}

    def scan(self) -> list:
        """Scan directories for .claude-plugin/plugin.json manifests."""
        found = []
        for search_dir in self.search_dirs:
            plugin_dir = Path(search_dir) / ".claude-plugin"
            manifest_path = plugin_dir / "plugin.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    name = manifest.get("name", plugin_dir.parent.name)
                    self.plugins[name] = manifest
                    found.append(name)
                except (json.JSONDecodeError, OSError) as e:
                    logging.error(f"[Plugin] Failed to load {manifest_path}: {e}")
        return found

    def get_mcp_servers(self) -> dict:
        """
        Extract MCP server configs from loaded plugins.
        Returns {server_name: {command, args, env}}.
        """
        servers = {}
        for plugin_name, manifest in self.plugins.items():
            for server_name, config in manifest.get("mcpServers", {}).items():
                servers[f"{plugin_name}__{server_name}"] = config
        return servers


class MCPToolRouter:
    """
    Routes tool calls to the correct MCP server.

    MCP tools are prefixed mcp__{server}__{tool} and live alongside
    native tools in the same tool pool. The router strips the prefix
    and dispatches to the right MCPClient.
    """
    def __init__(self):
        self.clients = {}

    def register_client(self, client: MCPClient):
        self.clients[client.server_name] = client

    def is_mcp_tool(self, tool_name: str) -> bool:
        return tool_name.startswith("mcp__")

    def call(self, tool_name: str, arguments: dict) -> str:
        """Route an MCP tool call to the correct server."""
        parts = tool_name.split("__", 2)
        if len(parts) != 3:
            return "Error: Invalid MCP tool name: {tool_name}"
        _, server_name, actual_tool = parts
        client = self.clients.get(server_name)
        if not client:
            return f"Error: MCP server not found: {server_name}"
        return client.call_tool(actual_tool, arguments)

    def get_all_tools(self) -> list:
        tools = []
        for client in self.clients.values():
            tools.extend(client.get_agent_tools())
        return tools


class CapabilityPermissionGate:
    """
    Shared permission gate for native tools and external capabilities.

    The teaching goal is simple: MCP does not bypass the control plane.
    Native tools and MCP tools both become normalized capability intents first,
    then pass through the same allow / ask policy.
    """
    READ_PREFIXES = ("read", "list", "get", "show", "search", "query", "inspect")
    HIGH_RISK_PREFIXES = ("delete", "remove", "drop", "shutdown")

    def __init__(self, mode: str = "default"):
        self.mode = mode if mode in PERMISSION_MODES else "default"

    def normalize(self, tool_name: str, tool_input: dict) -> dict:
        if tool_name.startswith("mcp__"):
            _, server_name, actual_tool = tool_name.split("__", 2)
            source = "mcp"
        else:
            server_name = None
            actual_tool = tool_name
            source = "native"
        lowered = actual_tool.lower()
        if actual_tool == "read_file" or lowered.startswith(self.READ_PREFIXES):
            risk = "read"
        elif actual_tool == "bash":
            command = tool_input.get("command", "")
            if any(token in command for token in ("rm -rf", "sudo", "shutdown", "reboot")):
                risk = "high"
            else:
                risk = "write"
        elif lowered.startswith(self.HIGH_RISK_PREFIXES):
            risk = "high"
        else:
            risk = "write"
        return {
            "source": source,
            "server": server_name,
            "tool": actual_tool,
            "risk": risk,
        }

    def check(self, tool_name: str, tool_input: dict) -> dict:
        intent = self.normalize(tool_name, tool_input)

        if intent["risk"] == "read":
            return {"behavior": "allow", "reason": "Read capability", "intent": intent}
        if self.mode == "auto" and intent["risk"] != "high":
            return {
                "behavior": "allow",
                "reason": "Auto mode for non-high-risk capability",
                "intent": intent,
            }
        if intent["risk"] == "high":
            return {
                "behavior": "ask",
                "reason": "High-risk capability requires confirmation",
                "intent": intent,
            }
        return {
            "behavior": "ask",
            "reason": "State-changing capability requires confirmation",
            "intent": intent,
        }

    def ask_user(self, intent: dict, tool_input: dict) -> bool:
        preview = json.dumps(tool_input, ensure_ascii=False)[:200]
        source = (
            f"{intent['source']}:{intent['server']}/{intent['tool']}"
            if intent.get("server")
            else f"{intent['source']}:{intent['tool']}"
        )
        print(f"\n  [Permission] {source} risk={intent['risk']}: {preview}")
        try:
            answer = input("  Allow? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")


# permission_gate = CapabilityPermissionGate()



# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    REQUEST_STORE.create({
        "request_id": req_id,
        "kind": "shutdown",
        "from": "lead",
        "to": teammate,
        "status": "pending",
        "create_at": time.time(),
        "updated_at": time.time(),
    })
    BUS.send("lead", teammate, "Please shut down gracefully.",
             "shutdown_request", {"request_id": req_id})
    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    req = REQUEST_STORE.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    REQUEST_STORE.update(
        request_id,
        status="approved" if approve else "rejected",
        reviewed_by="lead",
        resolved_at=time.time(),
        feedback=feedback,
    )
    BUS.send(
        "lead", req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {'approved' if approve else 'rejected'} for '{req['from']}'"


def _check_shutdown_status(request_id: str) -> str:
    return json.dumps(REQUEST_STORE.get(request_id) or {"error": "not found"})


memory_mgr = MemoryManager()


def run_save_memory(name: str, description: str, mem_type: str, content: str) -> str:
    return memory_mgr.save_memory(name, description, mem_type, content)


NATIVE_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "load_skill": lambda **kw: SKILL_REGISTRY.load_full_text(kw["name"]),
    "save_memory": lambda **kw: run_save_memory(kw["name"], kw["description"], kw["type"], kw["content"]),
    "todo": lambda **kw: TODO.update(kw["items"]),
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("owner"), kw.get("addBlockedBy"),
                                             kw.get("addBlocks")),
    "task_list": lambda **kw: TASKS.list_all(),
    "task_get": lambda **kw: TASKS.get(kw["task_id"]),
    "background_run": lambda **kw: BG.run(kw["command"]),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
    "cron_create": lambda **kw: scheduler.create(
        kw["cron"], kw["prompt"], kw.get("recurring", True), kw.get("durable", False)),
    "cron_delete": lambda **kw: scheduler.delete(kw["id"]),
    "cron_list": lambda **kw: scheduler.list_tasks(),
    "spawn_teammate": lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates": lambda **kw: TEAM.list_all(),
    "send_message": lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox": lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast": lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request": lambda **kw: handle_shutdown_request(kw["teammate"]),
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval": lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
    "idle": lambda **kw: "Lead does not idle.",
    "claim_task": lambda **kw: claim_task(kw["task_id"], "lead"),
    "task_bind_worktree": lambda **kw: TASKS.bind_worktree(kw["task_id"], kw["worktree"], kw.get("owner", "")),
    "worktree_create": lambda **kw: WORKTREES.create(kw["name"], kw.get("task_id"), kw.get("base_ref", "HEAD")),
    "worktree_list": lambda **kw: WORKTREES.list_all(),
    "worktree_enter": lambda **kw: WORKTREES.enter(kw["name"]),
    "worktree_status": lambda **kw: WORKTREES.status(kw["name"]),
    "worktree_run": lambda **kw: WORKTREES.run(kw["name"], kw["command"]),
    "worktree_closeout": lambda **kw: WORKTREES.closeout(
        kw["name"],
        kw["action"],
        kw.get("reason", ""),
        kw.get("force", False),
        kw.get("complete_task", False),
    ),
    "worktree_keep": lambda **kw: WORKTREES.keep(kw["name"]),
    "worktree_remove": lambda **kw: WORKTREES.remove(
        kw["name"],
        kw.get("force", False),
        kw.get("complete_task", False),
        kw.get("reason", ""),
    ),
    "worktree_events": lambda **kw: EVENTS.list_recent(kw.get("limit", 20)),
}

NATIVE_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"},
                                                       "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"name": "save_memory", "description": "Save a persistent memory that survives across sessions.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string", "description": "Short identifier (e.g. prefer_tabs, db_schema)"},
         "description": {"type": "string", "description": "One-line summary of what this memory captures"},
         "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"],
                  "description": "user=preferences, feedback=corrections, project=non-obvious project conventions or decision reasons, reference=external resource pointers"},
         "content": {"type": "string", "description": "Full memory content (multi-line OK)"},
     }, "required": ["name", "description", "type", "content"]}},
    {
        "name": "todo",
        "description": "Rewrite the current session plan for multi-step work.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Optional present-continuous label.",
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["items"],
        },
    },
    {"name": "task_create", "description": "Create a new task.",
     "input_schema": {"type": "object",
                      "properties": {"subject": {"type": "string"}, "description": {"type": "string"}},
                      "required": ["subject"]}},
    {"name": "task_update", "description": "Update a task's status, owner, or dependencies.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string",
                                                                                                  "enum": ["pending",
                                                                                                           "in_progress",
                                                                                                           "completed",
                                                                                                           "deleted"]},
                                                       "owner": {"type": "string",
                                                                 "description": "Set when a teammate claims the task"},
                                                       "addBlockedBy": {"type": "array", "items": {"type": "integer"}},
                                                       "addBlocks": {"type": "array", "items": {"type": "integer"}}},
                      "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks with status summary.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get full details of a task by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    {
        "name": "load_skill",
        "description": "Load the full body of a named skill into the current context.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {"name": "background_run", "description": "Run command in background thread. Returns task_id immediately.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "check_background", "description": "Check background task status. Omit task_id to list all.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "string"}}}},
    {"name": "cron_create", "description": "Schedule a recurring or one-shot task with a cron expression.",
     "input_schema": {"type": "object", "properties": {
         "cron": {"type": "string", "description": "5-field cron expression: 'min hour dom month dow'"},
         "prompt": {"type": "string", "description": "The prompt to inject when the task fires"},
         "recurring": {"type": "boolean", "description": "true=repeat, false=fire once then delete. Default true."},
         "durable": {"type": "boolean", "description": "true=persist to disk, false=session-only. Default false."},
     }, "required": ["cron", "prompt"]}},
    {"name": "cron_delete", "description": "Delete a scheduled task by ID.",
     "input_schema": {"type": "object", "properties": {
         "id": {"type": "string", "description": "Task ID to delete"},
     }, "required": ["id"]}},
    {"name": "cron_list", "description": "List all scheduled tasks.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "spawn_teammate", "description": "Spawn a persistent teammate that runs in its own thread.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"},
                                                       "prompt": {"type": "string"}},
                      "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates with name, role, status.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a teammate's inbox.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"},
                                                       "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}},
                      "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": []}},
    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
    {"name": "shutdown_request",
     "description": "Request a teammate to shut down gracefully. Returns a request_id for tracking.",
     "input_schema": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}},
    {"name": "shutdown_response", "description": "Check the status of a shutdown request by request_id.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}}, "required": ["request_id"]}},
    {"name": "plan_approval",
     "description": "Approve or reject a teammate's plan. Provide request_id + approve + optional feedback.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"},
                                                       "feedback": {"type": "string"}},
                      "required": ["request_id", "approve"]}},
    {"name": "idle", "description": "Enter idle state (for lead -- rarely used).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "claim_task", "description": "Claim a task from the board by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
    {"name": "task_bind_worktree", "description": "Bind a task to a worktree name.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "worktree": {"type": "string"},
                                                       "owner": {"type": "string"}},
                      "required": ["task_id", "worktree"]}},
    {"name": "worktree_create", "description": "Create a git worktree and optionally bind it to a task.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "task_id": {"type": "integer"},
                                                       "base_ref": {"type": "string"}}, "required": ["name"]}},
    {"name": "worktree_list", "description": "List worktrees tracked in .worktrees/index.json.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "worktree_enter", "description": "Enter or reopen a worktree lane before working in it.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "worktree_status", "description": "Show git status for one worktree.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "worktree_run", "description": "Run a shell command in a named worktree directory.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "command": {"type": "string"}},
                      "required": ["name", "command"]}},
    {"name": "worktree_closeout", "description": "Close out a lane by keeping it for follow-up or removing it.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"},
                                                       "action": {"type": "string", "enum": ["keep", "remove"]},
                                                       "reason": {"type": "string"}, "force": {"type": "boolean"},
                                                       "complete_task": {"type": "boolean"}},
                      "required": ["name", "action"]}},
    {"name": "worktree_remove", "description": "Remove a worktree and optionally mark its bound task completed.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "force": {"type": "boolean"},
                                                       "complete_task": {"type": "boolean"},
                                                       "reason": {"type": "string"}}, "required": ["name"]}},
    {"name": "worktree_keep", "description": "Mark a worktree as kept without removing it.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "worktree_events", "description": "List recent lifecycle events.",
     "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},

]

MEMORY_GUIDANCE = """
When to save memories:
- User states a preference ("I like tabs", "always use pytest") -> type: user
- User corrects you ("don't do X", "that was wrong because...") -> type: feedback
- You learn a project fact that is not easy to infer from current code alone
  (for example: a rule exists because of compliance, or a legacy module must
  stay untouched for business reasons) -> type: project
- You learn where an external resource lives (ticket board, dashboard, docs URL)
  -> type: reference

When NOT to save:
- Anything easily derivable from code (function signatures, file structure, directory layout)
- Temporary task state (current branch, open PR numbers, current TODOs)
- Secrets or credentials (API keys, passwords)
"""



# -- MCP Tool Router (global) --
mcp_router = MCPToolRouter()
plugin_loader = PluginLoader()


def build_tool_pool() -> list:
    """
    Assemble the complete tool pool: native + MCP tools.

    Native tools take precedence on name conflicts so the local core remains
    predictable even after external tools are added.
    :return:
    """
    all_tools = list(NATIVE_TOOLS)
    mcp_tools = mcp_router.get_all_tools()

    native_names = {t["name"] for t in all_tools}
    for tool in mcp_tools:
        if tool["name"] not in native_names:
            all_tools.append(tool)

    return all_tools


def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    if mcp_router.is_mcp_tool(tool_name):
        return mcp_router.call(tool_name, tool_input)
    handler = NATIVE_HANDLERS.get(tool_name)
    if handler:
        return handler(**tool_input)
    return f"Unknown tool: {tool_name}"


def normalize_tool_result(output: str, intent: dict) -> str:
    status = "error" if "Error:" in output else "ok"
    payload = {
        "source": intent["source"],
        "server": intent.get("server"),
        "tool": intent["tool"],
        "risk": intent["risk"],
        "status": status,
        "preview": output[:500]
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def permission_execute_tool(block, results, permission_gate: CapabilityPermissionGate, hooks: HookManager):
    decision = permission_gate.check(block.name, block.input or {})
    if decision["behavior"] == "deny":
        output = f"Permission denied: {decision['reason']}"
    elif decision["behavior"] == "ask" and not permission_gate.ask_user(
            decision["intent"], block.input or {}
    ):
        output = f"Permission denied by user: {decision['reason']}"
    else:
        output = hook_and_tool(block, results, hooks, decision)

    results.append({
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": str(output),
    })


def pre_hooks(results: list, ctx: dict, hooks: HookManager, ):
    # -- PreToolUse hooks --
    pre_result = hooks.run_hooks("PreToolUse", ctx)
    block.input = ctx.get("tool_input")

    # Inject hook messages into results
    for msg in pre_result.get("messages", []):
        results.append({
            "type": "tool_result", "tool_use_id": block.id,
            "content": f"[Hook message]: {msg}",
        })

    if pre_result.get("blocked"):
        reason = pre_result.get("block_reason", "Blocked by hook")
        output = f"Tool blocked by PreToolUse hook: {reason}"
        return output


def post_hooks(ctx: dict, output: str, hooks: HookManager, ):
    # -- PostToolUse hooks --
    ctx["tool_output"] = output
    post_result = hooks.run_hooks("PostToolUse", ctx)

    # Inject post-hook messages
    for msg in post_result.get("messages", []):
        output += f"\n[Hook note]: {msg}"
    return output


def hook_and_tool(block, results, hooks: HookManager, decision: dict):
    tool_input = dict(block.input or {})
    ctx = {"tool_name": block.name, "tool_input": tool_input}

    pre_hooks_res = pre_hooks(results, ctx, hooks)
    if pre_hooks_res:
        return

    try:
        output = handle_tool_call(block.name, block.input or {})
    except Exception as e:
        output = f"Error: {e}"
    logging.info(f"> Tool ouput---{block.name}: {str(output)[:200]}")

    output = post_hooks(ctx, output, hooks)

    output = normalize_tool_result(str(output), decision.get("intent"))
    return output


prompt_builder = SystemPromptBuilder(workdir=WORKDIR, tools=build_tool_pool())


def agent_loop(messages: list, hooks: HookManager, perms: CapabilityPermissionGate):
    """
    Agent loop with assembled system prompt.

    The system prompt is rebuilt each iteration. In real CC, the static
    prefix is cached and only the dynamic suffix changes per turn.
    """
    max_output_recovery_count = 0
    tools = build_tool_pool()
    while True:
        system = prompt_builder.build()
        notifs = BG.drain_notifications()
        if notifs and messages:
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['status']}: {n['preview']} "
                f"(output_file={n['output_file']})"
                for n in notifs
            )
            messages.append({"role": "user", "content": f"<background-results>\n{notif_text}\n</background-results>"})

        notifications = scheduler.drain_notifications()
        for note in notifications:
            logging.info(f"[Cron notification] {note[:200]}")
            messages.append({"role": "user", "content": note})

        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>"
            })
            messages.append({
                "role": "assistant",
                "content": "Noted inbox messages",
            })

        response = None
        for attempt in range(MAX_RECOVERY_ATTEMPTS + 1):
            try:
                response = client.messages.create(
                    model=MODEL, system=system, messages=messages,
                    tools=tools, max_tokens=8000,
                )
                break
            except APIError as e:
                logging.info(f"API error--------messages: {messages}")
                error_body = str(e).lower()
                # Strategy 2: prompt_too_long -> compact and retry
                if "overlong_prompt" in error_body or ("prompt" in error_body and "long" in error_body):
                    logging.info(f"[Recovery] Prompt too long. Compacting... (attempt {attempt + 1})")
                    messages[:] = auto_compact(messages)
                    continue

                # Strategy 3: connection/rate errors -> backoff
                if attempt < MAX_RECOVERY_ATTEMPTS:
                    delay = backoff_delay(attempt)
                    logging.info(f"[Recovery] API error: {e}. "
                                 f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RECOVERY_ATTEMPTS})")
                    time.sleep(delay)
                    continue
                # All retries exhausted
                logging.info(f"[Error] API call failed after {MAX_RECOVERY_ATTEMPTS} retries: {e}")
                return
            except (ConnectionError, TimeoutError, OSError) as e:
                if attempt < MAX_RECOVERY_ATTEMPTS:
                    delay = backoff_delay(attempt)
                    logging.info(f"[Recovery] Connection error: {e}. "
                                 f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RECOVERY_ATTEMPTS})")
                    time.sleep(delay)
                    continue
                logging.info(f"[Error] Connection failed after {MAX_RECOVERY_ATTEMPTS} retries: {e}")
                return
        if response is None:
            logging.info("[Error] No response received.")
            return

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "max_tokens":
            max_output_recovery_count += 1
            if max_output_recovery_count <= MAX_RECOVERY_ATTEMPTS:
                logging.info(f"[Recovery] max_tokens hit "
                             f"({max_output_recovery_count}/{MAX_RECOVERY_ATTEMPTS}). "
                             "Injecting continuation...")
                messages.append({"role": "user", "content": CONTINUATION_MESSAGE})
                continue
            else:
                logging.info(f"[Error] max_tokens recovery exhausted "
                             f"({MAX_RECOVERY_ATTEMPTS} attempts). Stopping.")
                return
        # Reset max_tokens counter on successful non-max_tokens response
        max_output_recovery_count = 0

        if response.stop_reason != "tool_use":
            return

        results = []
        used_todo = False
        for block in response.content:
            if block.type != "tool_use":
                continue

            permission_execute_tool(block, results, perms, hooks)

            if block.name == "todo":
                used_todo = True
        if used_todo:
            TODO.state.rounds_since_update = 0
        else:
            TODO.note_round_without_update()
            reminder = TODO.reminder()
            if reminder:
                # results.insert(0, {"type": "text", "text": reminder})
                results.append({"type": "text", "text": reminder})

        messages.append({"role": "user", "content": results})

        # Check if we should auto-compact (proactive, not just reactive)
        if estimate_tokens(messages) > TOKEN_THRESHOLD:
            logging.info("[Recovery] Token estimate exceeds threshold. Auto-compacting...")
            messages[:] = auto_compact(messages)


if __name__ == "__main__":
    logging.info("Permission modes: default, plan, auto\n "
                 "[Error recovery enabled: max_tokens / prompt_too_long / connection backoff]")
    mode_input = input("Mode (default): ").strip().lower() or "default"
    if mode_input not in MODES:
        mode_input = "default"

    # perms = PermissionManager(mode=mode_input)
    perms = CapabilityPermissionGate(mode=mode_input)
    logging.info(f"[Permission mode: {mode_input}]")

    full_prompt = prompt_builder.build()
    section_count = full_prompt.count("\n#")
    logging.info(f"[System prompt assembled: {len(full_prompt)} chars, ~{section_count} sections]")

    hooks = HookManager()
    hooks.run_hooks("SessionStart", {"tool_name": "", "tool_input": {}})

    scheduler.start()

    # Scan for plugins
    found = plugin_loader.scan()
    if found:
        print(f"[Plugins loaded: {', '.join(found)}]")
        for server_name, config in plugin_loader.get_mcp_servers().items():
            mcp_client = MCPClient(server_name, config.get("command", ""), config.get("args", []))
            if mcp_client.connect():
                mcp_client.list_tools()
                mcp_router.register_client(mcp_client)
                print(f"[MCP] Connected to {server_name}")

    tool_count = len(build_tool_pool())
    mcp_count = len(mcp_router.get_all_tools())
    print(f"[Tool pool: {tool_count} tools ({mcp_count} from MCP)]")


    history = []
    while True:

        try:
            query = input("\033[36ms14 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            scheduler.stop()
            continue
        if not query:
            continue

        if query.strip().lower() in ("q", "exit", ""):
            scheduler.stop()
            break

        if query.strip().lower() == "/cron":
            print(scheduler.list_tasks())
            continue

        if query.strip().lower() == "/team":
            print(TEAM.list_all())
            continue

        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue

        if query.strip() == "/tasks":
            TASKS_DIR.mkdir(exist_ok=True)
            for f in sorted(TASKS_DIR.glob("task_*.json")):
                t = json.loads(f.read_text())
                marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
                owner = f" @{t['owner']}" if t.get("owner") else ""
                print(f"  {marker} #{t['id']}: {t['subject']}{owner}")
            continue

        # /mode command to switch modes at runtime
        if query.startswith("/mode"):
            parts = query.split()
            if len(parts) == 2 and parts[1] in MODES:
                perms.mode = parts[1]
                print(f"[Switched to {parts[1]} mode]")
            else:
                print(f"Usage: /mode <{'|'.join(MODES)}>")
            continue

        if query.strip() == "/tools":
            for tool in build_tool_pool():
                prefix = "[MCP] " if tool["name"].startswith("mcp__") else "       "
                print(f"  {prefix}{tool['name']}: {tool.get('description', '')[:60]}")
            continue

        if query.strip() == "/mcp":
            if mcp_router.clients:
                for name, c in mcp_router.clients.items():
                    tools = c.get_agent_tools()
                    print(f"  {name}: {len(tools)} tools")
            else:
                print("  (no MCP servers connected)")
            continue


        # # /rules command to show current rules
        # if query.strip() == "/rules":
        #     for i, rule in enumerate(perms.rules):
        #         print(f"  {i}: {rule}")
        #     continue

        if query.strip().lower() == "/prompt":
            print("---- System prompt ----")
            print(prompt_builder.build())
            print("---- End prompt ----")
            continue
        if query.strip().lower() == "/sections":
            prompt = prompt_builder.build()
            for line in prompt.splitlines():
                if line.startswith("# ") or line == DYNAMIC_BOUNDARY:
                    print(f"  {line}")
            continue

        history.append({"role": "user", "content": query})
        agent_loop(history, hooks, perms)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    logging.info(block.text)
        logging.info("")

    for c in mcp_router.clients.values():
        c.disconnect()