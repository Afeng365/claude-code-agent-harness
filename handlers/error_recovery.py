import json
from random import random

from dao.anthropic_utils import client
from settings.constant import MODEL, BACKOFF_BASE_DELAY, BACKOFF_MAX_DELAY


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