import json
import subprocess
import threading
import time
import uuid
from pathlib import Path

from dao.anthropic_utils import client
from handlers.compact_context import maybe_persist_output
from log import logging
from settings.constant import VALID_MSG_TYPES, WORKDIR, IDLE_TIMEOUT, POLL_INTERVAL, MODEL, TASKS_DIR, \
    CLAIM_EVENTS_PATH, INBOX_DIR, TEAM_DIR, REQUESTS_DIR, PERSIST_OUTPUT_TRIGGER_CHARS_BASH

_claim_lock = threading.Lock()


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


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str, tool_use_id: str = "") -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        if not out:
            return "(no output)"
        out = maybe_persist_output(tool_use_id, out, trigger_chars=PERSIST_OUTPUT_TRIGGER_CHARS_BASH)
        # return out[:CONTEXT_TRUNCATE_CHARS] if isinstance(out, str) else str(out)[:CONTEXT_TRUNCATE_CHARS]
        return out
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None, tool_use_id: str = "") -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        out = "\n".join(lines)
        out = maybe_persist_output(tool_use_id, out)
        # return out[:CONTEXT_TRUNCATE_CHARS] if isinstance(out, str) else str(out)[:CONTEXT_TRUNCATE_CHARS]
        return out
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
            "claim_task": lambda **kw: claim_task(kw["task_id"], kw["sender"],
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


BUS = MessageBus(INBOX_DIR)
TEAM = TeammateManager(TEAM_DIR)
REQUEST_STORE = RequestStore(REQUESTS_DIR)
