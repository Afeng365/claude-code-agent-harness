#!/usr/bin/env python3
# Harness: autonomy -- models that find work without being told.

import json
import time
from anthropic import APIError

from dao.anthropic_utils import client
from handlers.background_tasks import BG
from handlers.compact_context import CompactState, compact_history, micro_compact, estimate_context_size
from handlers.cron_scheduler import scheduler
from handlers.error_recovery import auto_compact, backoff_delay, estimate_tokens
from handlers.hook_system import HookManager, hooks
from log import logging
from handlers.mcp_plugin import mcp_router, plugin_loader, MCPClient
from handlers.permission_system import CapabilityPermissionGate
from services.agent_service import build_tool_pool, permission_execute_tool
from settings.constant import MODEL, MODES, TASKS_DIR, DYNAMIC_BOUNDARY, \
    MAX_RECOVERY_ATTEMPTS, CONTEXT_LIMIT, CONTINUATION_MESSAGE, TOKEN_THRESHOLD, WORKDIR
from handlers.system_prompt import SystemPromptBuilder
from handlers.team_system import TEAM, BUS
from handlers.todomanager import TODO


prompt_builder = SystemPromptBuilder(workdir=WORKDIR, tools=build_tool_pool())



def agent_loop(messages: list, hooks: HookManager, perms: CapabilityPermissionGate, state: CompactState):
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

        messages[:] = micro_compact(messages)

        if estimate_context_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages, state)

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
        manual_compact = False
        compact_focus = None
        for block in response.content:
            if block.type != "tool_use":
                continue

            permission_execute_tool(block, results, perms, hooks)

            if block.name == "todo":
                used_todo = True
            if block.name == "compact":
                manual_compact = True
                compact_focus = (block.input or {}).get("focus")
        if used_todo:
            TODO.state.rounds_since_update = 0
        else:
            TODO.note_round_without_update()
            reminder = TODO.reminder()
            if reminder:
                # results.insert(0, {"type": "text", "text": reminder})
                results.append({"type": "text", "text": reminder})

        messages.append({"role": "user", "content": results})

        if manual_compact:
            print("[manual compact]")
            messages[:] = compact_history(messages, state, focus=compact_focus)

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

    compact_state = CompactState()
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
        agent_loop(history, hooks, perms, compact_state)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    logging.info(block.text)
        logging.info("")

    for c in mcp_router.clients.values():
        c.disconnect()
