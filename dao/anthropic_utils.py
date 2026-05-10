import os

from anthropic import Anthropic

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))