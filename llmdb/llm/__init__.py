"""
LLM integration helpers.

Provides a thin, dependency-free client for OpenAI-compatible chat-completion
APIs, plus an optional schema-aware prompt builder used by higher-level
components such as the NaturalLanguageTranslator.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChatMessage:
    role: str    # system | user | assistant
    content: str


class LLMClient:
    """Minimal OpenAI-compatible chat-completion client.

    Uses only the Python standard library (urllib) — no third-party packages
    required.  Supports any API that implements the /chat/completions endpoint.

    Parameters
    ----------
    api_key  : Bearer token for the API.
    model    : Model identifier (e.g. "gpt-4o-mini", "llama-3-70b-instruct").
    base_url : Root of the API (default: OpenAI).
    timeout  : Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------

    def complete(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat-completion request and return the assistant's text."""
        payload = json.dumps({
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = json.loads(resp.read())
        return body["choices"][0]["message"]["content"]

    def complete_json(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Any:
        """Same as complete() but parses and returns the JSON body."""
        text = self.complete(messages, temperature, max_tokens)
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.rstrip("`").strip()
        return json.loads(text)


# ---------------------------------------------------------------------------
# Schema prompt builder
# ---------------------------------------------------------------------------

def schema_to_prompt(schema_context: Dict[str, Any]) -> str:
    """Convert a schema context dict into a compact, LLM-readable text block."""
    lines = ["Database schema:"]
    for table_name, info in schema_context.get("tables", {}).items():
        cols = ", ".join(
            f"{c['name']} {c['type']}"
            + (" PK" if c.get("primary_key") else "")
            + ("?" if c.get("nullable", True) else "")
            for c in info.get("columns", [])
        )
        lines.append(f"  TABLE {table_name} ({cols})")
    return "\n".join(lines)
