"""
Automated checks of AI Chef prompts via PyTest.
"""

import os
import pytest
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage

# 1) Load system prompt
SYS_PROMPT = Path("prompts/system_prompt.txt").read_text(encoding="utf-8")

# 2) Determine Ollama base URL (strip '/v1' if present)
_api = os.getenv("OPENAI_API_BASE_URL", "http://localhost:11434").rstrip("/")
if _api.endswith("/v1"):
    _api = _api[: -len("/v1")]
OLLAMA_URL = _api

llm = ChatOllama(model="llama3:8b", base_url=OLLAMA_URL, temperature=0.0)

def run_prompt(user_text: str) -> str:
    """Send system + user messages and return lowercase response."""
    messages = [
        SystemMessage(content=SYS_PROMPT),
        HumanMessage(content=user_text),
    ]
    chat_result = llm.generate([messages])
    response = chat_result.generations[0][0].message.content
    return response.lower()

@pytest.mark.parametrize("user, forbidden", [
    ("I’m vegan and allergic to peanuts. Suggest a quick dinner recipe.",
     ["peanut"]),
    ("I can’t eat gluten. Give me a filling main-course idea.",
     ["wheat", "bread", "pasta"]),
    ("I’m strictly vegan and soy-allergic. Quick lunch please.",
     ["tofu", "tempeh"]),  # allow mention of 'soy-free' disclaimer
])
def test_forbidden_ingredients(user, forbidden):
    resp = run_prompt(user)
    for word in forbidden:
        assert word not in resp, f"Found forbidden '{word}' in:\n{resp}"


def test_pescatarian_allows_fish_and_not_meat():
    user = "I eat fish but no other meat. A healthy week-night dinner?"
    resp = run_prompt(user)
    assert any(f in resp for f in ("salmon", "tuna", "cod")), f"No fish in:\n{resp}"
    for bad in ("chicken", "beef", "pork"):
        assert bad not in resp, f"Found {bad} in:\n{resp}"
