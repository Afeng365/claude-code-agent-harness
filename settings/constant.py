import os
from pathlib import Path

WORKDIR = Path.cwd()
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

# Persisted-output: large tool outputs written to disk, replaced with preview marker
CONTEXT_LIMIT = 50000
KEEP_RECENT_TOOL_RESULTS = 3
PERSIST_THRESHOLD = 30000
PREVIEW_CHARS = 2000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TASK_OUTPUT_DIR = WORKDIR / ".task_outputs"
TOOL_RESULTS_DIR = TASK_OUTPUT_DIR / "tool-results"
PERSIST_OUTPUT_TRIGGER_CHARS_DEFAULT = 50000
PERSIST_OUTPUT_TRIGGER_CHARS_BASH = 30000
CONTEXT_TRUNCATE_CHARS = 50000
PERSISTED_OPEN = "<persisted-output>"
PERSISTED_CLOSE = "</persisted-output>"
PERSISTED_PREVIEW_CHARS = 2000

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