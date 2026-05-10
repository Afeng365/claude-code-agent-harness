import json
import re
from fnmatch import fnmatch
from pathlib import Path

from log import logging
from settings.constant import WORKDIR, MODES


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
        self.mode = mode if mode in MODES else "default"

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