import os

from anthropic import Anthropic
from langsmith.wrappers import wrap_anthropic

client = wrap_anthropic(Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL")))

