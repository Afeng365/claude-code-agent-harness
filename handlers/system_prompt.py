import platform
import re
from datetime import datetime
from pathlib import Path

from settings.constant import MODEL, WORKDIR, DYNAMIC_BOUNDARY
from handlers.skill_loading import SKILL_REGISTRY

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

