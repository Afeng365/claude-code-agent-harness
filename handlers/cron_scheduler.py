import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue, Empty

from log import logging
from settings.constant import CRON_LOCK_FILE, SCHEDULED_TASKS_FILE, JITTER_MINUTES, JITTER_OFFSET_MAX, AUTO_EXPIRY_DAYS


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
