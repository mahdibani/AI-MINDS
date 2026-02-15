"""
memory.py – Mem0 memory layer wrapper.

Provides a unified Memory interface backed by mem0ai (self-hosted).
Configured to use the same local Ollama instance as the rest of AI-MINDS.

Falls back to a simple in-process dict store if mem0 is unavailable,
so the rest of the system never crashes due to missing memory.

Usage:
    from rlm.memory import AgentMemory
    mem = AgentMemory(user_id="bani")
    mem.add("User prefers CSV files for budget analysis")
    results = mem.search("budget")
    mem.inject_into_messages(messages)   # prepends relevant memories
"""

from __future__ import annotations

import os
import json
import datetime
from typing import Any, Dict, List, Optional


# ── Graceful import ────────────────────────────────────────────────────────────

try:
    from mem0 import Memory as _Mem0Memory
    _MEM0_AVAILABLE = True
except ImportError:
    _Mem0Memory = None
    _MEM0_AVAILABLE = False


# ── Fallback in-process store ──────────────────────────────────────────────────

class _InMemoryStore:
    """Minimal mem0-compatible store used when mem0ai is not installed."""

    def __init__(self):
        self._store: List[Dict[str, Any]] = []

    def add(self, messages, user_id="default", **_):
        text = messages if isinstance(messages, str) else json.dumps(messages)
        self._store.append({
            "memory": text,
            "user_id": user_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        })
        return {"results": [{"memory": text}]}

    def search(self, query, user_id="default", limit=5, **_):
        # Simple keyword match
        q = query.lower()
        hits = [e for e in self._store
                if e["user_id"] == user_id and q in e["memory"].lower()]
        return {"results": hits[-limit:]}

    def get_all(self, user_id="default", **_):
        hits = [e for e in self._store if e["user_id"] == user_id]
        return {"results": hits}

    def delete_all(self, user_id="default", **_):
        self._store = [e for e in self._store if e["user_id"] != user_id]


# ── Main wrapper ───────────────────────────────────────────────────────────────

class AgentMemory:
    """
    Unified memory interface for AI-MINDS agents.

    Backed by mem0 (self-hosted with Ollama) when available,
    falls back to an in-process store otherwise.

    Key methods:
        add(text_or_messages, metadata={})  – store a memory
        search(query, limit=5)              – retrieve relevant memories
        get_all()                           – dump all memories for this user
        inject_into_messages(messages)      – prepend memories to a chat list
        clear()                             – wipe all memories for this user
    """

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self._backend_name = "unknown"
        self._mem = self._init_backend()

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_backend(self):
        if not _MEM0_AVAILABLE:
            print("[memory] mem0ai not installed – using in-process fallback store")
            self._backend_name = "in-process"
            return _InMemoryStore()

        ollama_base = os.getenv("RLM_API_URL", "http://localhost:11434/v1").rstrip("/v1").rstrip("/")
        llm_model   = os.getenv("RLM_ROOT_MODEL") or os.getenv("RLM_WORKER_MODEL", "qwen2.5:3b")
        embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model":    llm_model,
                    "ollama_base_url": ollama_base,
                    "temperature": 0.1,
                    "max_tokens": 2000,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model":    embed_model,
                    "ollama_base_url": ollama_base,
                },
            },
            # Use SQLite vector store so no extra infra is needed
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "ai_minds_memories",
                    "path":            "./data/mem0_qdrant",
                    "on_disk":         True,
                },
            },
        }

        try:
            mem = _Mem0Memory.from_config(config)
            self._backend_name = f"mem0+ollama({llm_model})"
            print(f"[memory] mem0 initialised → {self._backend_name}")
            return mem
        except Exception as e:
            print(f"[memory] mem0 init failed ({e}) – falling back to in-process store")
            self._backend_name = "in-process"
            return _InMemoryStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, content, metadata: Optional[Dict] = None) -> Dict:
        """
        Store a memory.

        content can be:
          - str: a plain text fact or observation
          - list: a chat message list [{"role": ..., "content": ...}, ...]
        """
        try:
            if isinstance(content, str):
                messages = [{"role": "user", "content": content}]
            else:
                messages = content
            kwargs = {"user_id": self.user_id}
            if metadata:
                kwargs["metadata"] = metadata
            result = self._mem.add(messages, **kwargs)
            return result
        except Exception as e:
            print(f"[memory] add() failed: {e}")
            return {}

    def search(self, query: str, limit: int = 5) -> List[str]:
        """
        Search memories relevant to query.
        Returns a list of plain-text memory strings.
        """
        try:
            result = self._mem.search(query, user_id=self.user_id, limit=limit)
            entries = result.get("results", [])
            return [e.get("memory", str(e)) for e in entries]
        except Exception as e:
            print(f"[memory] search() failed: {e}")
            return []

    def get_all(self) -> List[Dict]:
        """Return all memories for this user."""
        try:
            result = self._mem.get_all(user_id=self.user_id)
            return result.get("results", [])
        except Exception as e:
            print(f"[memory] get_all() failed: {e}")
            return []

    def inject_into_messages(
        self,
        messages: List[Dict[str, str]],
        query: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Prepend relevant memories as a system context block.
        If query is provided, does a semantic search; otherwise fetches recent memories.

        Returns the augmented messages list (the original is not mutated).
        """
        try:
            if query:
                mems = self.search(query, limit=limit)
            else:
                all_mems = self.get_all()
                mems = [e.get("memory", str(e)) for e in all_mems[-limit:]]

            if not mems:
                return messages

            memory_block = "Relevant memories from previous sessions:\n" + \
                           "\n".join(f"- {m}" for m in mems)

            # Insert after any existing system message, or at position 0
            augmented = list(messages)
            sys_idx = next((i for i, m in enumerate(augmented) if m.get("role") == "system"), -1)
            insert_at = sys_idx + 1 if sys_idx >= 0 else 0
            augmented.insert(insert_at, {"role": "system", "content": memory_block})
            return augmented
        except Exception as e:
            print(f"[memory] inject_into_messages() failed: {e}")
            return messages

    def clear(self):
        """Delete all memories for this user."""
        try:
            self._mem.delete_all(user_id=self.user_id)
            print(f"[memory] Cleared all memories for user '{self.user_id}'")
        except Exception as e:
            print(f"[memory] clear() failed: {e}")

    def __repr__(self):
        return f"AgentMemory(user_id={self.user_id!r}, backend={self._backend_name!r})"