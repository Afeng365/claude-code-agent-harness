import json

from handlers.hook_system import HookManager
from log import logging
from handlers.mcp_plugin import mcp_router
from handlers.permission_system import CapabilityPermissionGate
from tools import NATIVE_TOOLS, NATIVE_HANDLERS


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


def pre_hooks(results: list, ctx: dict, hooks: HookManager, block):
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
    pre_hooks_res = pre_hooks(results, ctx, hooks, block)
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