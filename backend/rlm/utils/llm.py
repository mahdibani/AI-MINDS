"""
LLM client wrapper with cost tracking.

Reads configuration from environment / .env:
    RLM_API_URL      Base URL of the OpenAI-compatible server   (default: http://localhost:11434/v1)
    RLM_API_KEY      API key (also accepts OPENAI_API_KEY)
    RLM_ROOT_MODEL   Default root model name                     (default: qwen2.5:3b)
    RLM_WORKER_MODEL Default worker/sub-agent model name         (default: phi4-mini:latest)
"""

from __future__ import annotations

import os
import json
import requests
from typing import Any, Dict, List, Optional, Tuple


def get_llm_client(api_key: Optional[str], model: str) -> "LocalClient":
    """Factory: return an LLM client for the given model."""
    return LocalClient(api_key, model)


class LocalClient:
    """
    OpenAI-compatible local LLM client (Ollama).

    Tries /v1/chat/completions first; falls back to /v1/completions.
    Tracks approximate token counts and cost.
    """

    # Pricing per 1 M tokens (input_price, output_price) in USD
    # Local models are essentially free, but we track for visibility
    PRICING: Dict[str, Tuple[float, float]] = {
        # Ollama local models - treated as nearly free
        "qwen2.5:3b":       (0.01, 0.01),
        "qwen:4b":          (0.01, 0.01),
        "phi4-mini:latest": (0.01, 0.01),
        "gemma3:latest":    (0.01, 0.01),
        "default":          (0.01, 0.01),
    }

    def __init__(self, api_key: Optional[str], model: str):
        self.api_key  = api_key or os.getenv("RLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        # Ollama default port is 11434, not 8080
        self.base_url = os.getenv("RLM_API_URL", "http://localhost:11434/v1").rstrip("/")

        # Discover available models from the server
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach LLM server at {self.base_url}. "
                f"Check RLM_API_URL in your .env file. Error: {e}"
            )

        # Parse model list (handle OpenAI format and plain arrays)
        available: List[str] = []
        if isinstance(data, dict):
            items = data.get("data") or data.get("models") or []
            for item in items:
                mid = item.get("id") or item.get("model") or item.get("name") or ""
                if mid:
                    available.append(mid)
        elif isinstance(data, list):
            available = [str(m) for m in data]

        print(f"[llm] Available models: {available}")

        # Resolve model name
        if not model or model in ("", "auto"):
            if available:
                self.model = available[0]
                print(f"[llm] Auto-selected model: {self.model}")
            else:
                # Fallback to qwen2.5:3b if nothing available
                self.model = "qwen2.5:3b"
                print(f"[llm] No models reported, using fallback: {self.model}")
        elif model and model not in available and available:
            print(f"[llm] Warning: model '{model}' not found in {available}.")
            # Try to find a similar model or use first available
            fallback = self._find_fallback_model(model, available)
            if fallback:
                print(f"[llm] Using fallback: {fallback}")
                self.model = fallback
            else:
                print(f"[llm] Using requested model anyway: {model}")
                self.model = model
        else:
            self.model = model

    def _find_fallback_model(self, requested: str, available: List[str]) -> Optional[str]:
        """Find a fallback model based on requested name."""
        requested_lower = requested.lower()
        
        # Try to match by family
        if "qwen" in requested_lower:
            for m in available:
                if "qwen" in m.lower():
                    return m
        if "phi" in requested_lower:
            for m in available:
                if "phi" in m.lower():
                    return m
        if "gemma" in requested_lower:
            for m in available:
                if "gemma" in m.lower():
                    return m
        
        # Return first available as last resort
        return available[0] if available else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def completion(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Simple completion (no cost tracking)."""
        raw = self._make_request(messages, **kwargs)
        return self._extract_content(raw)

    def completion_with_cost(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Completion with approximate cost tracking."""
        raw = self._make_request(messages, **kwargs)
        content = self._extract_content(raw)

        # Use server-reported usage if available, else estimate
        usage = raw.get("usage", {})
        input_tokens  = usage.get("prompt_tokens")    or sum(len(m.get("content") or "") for m in messages) // 4
        output_tokens = usage.get("completion_tokens") or len(content) // 4
        total_tokens  = input_tokens + output_tokens

        ip, op = self.PRICING.get(self.model, self.PRICING["default"])
        cost = (input_tokens / 1_000_000 * ip) + (output_tokens / 1_000_000 * op)

        cost_info = {
            "cost":          cost,
            "tokens":        total_tokens,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
        }
        return content, cost_info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_request(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Make a request; try chat/completions first, fall back to completions."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # --- chat/completions ---
        try:
            payload: Dict[str, Any] = {
                "model":       self.model,
                "messages":    messages,
                "max_tokens":  kwargs.get("max_tokens",  2048),
                "temperature": kwargs.get("temperature", 0.7),
            }
            # Forward tool schemas if provided
            if "tools" in kwargs:
                payload["tools"]       = kwargs["tools"]
                payload["tool_choice"] = kwargs.get("tool_choice", "auto")

            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()
            print(f"[llm] chat/completions returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[llm] chat/completions exception: {e}")

        # --- completions fallback ---
        prompt = ""
        for m in messages:
            role    = m.get("role", "user").capitalize()
            content = m.get("content") or ""
            prompt += f"{role}: {content}\n"
        prompt += "Assistant: "

        payload2: Dict[str, Any] = {
            "model":       self.model,
            "prompt":      prompt,
            "max_tokens":  kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
            "stop":        kwargs.get("stop", ["User:", "Assistant:"]),
        }
        resp2 = requests.post(
            f"{self.base_url}/completions",
            headers=headers,
            json=payload2,
            timeout=120,
        )
        if resp2.status_code != 200:
            raise RuntimeError(f"LLM API error {resp2.status_code}: {resp2.text[:500]}")
        return resp2.json()

    @staticmethod
    def _extract_content(raw: Dict[str, Any]) -> str:
        """Pull the text content out of a chat or completion response."""
        choices = raw.get("choices", [])
        if not choices:
            raise ValueError(f"No choices in response: {raw}")
        choice = choices[0]
        if "message" in choice:
            return choice["message"].get("content") or ""
        if "text" in choice:
            return choice["text"]
        raise ValueError(f"Unexpected choice format: {choice}")