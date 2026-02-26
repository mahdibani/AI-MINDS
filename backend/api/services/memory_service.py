from __future__ import annotations

from api.core.dependencies import container


def add_memory(user_id: str, text: str, metadata: dict | None) -> str:
    mem = container.get_memory(user_id)
    mem.add(text, metadata=metadata)
    return mem._backend_name


def search_memory(user_id: str, query: str, limit: int) -> list:
    mem = container.get_memory(user_id)
    return mem.search(query, limit=limit)


def list_memory(user_id: str) -> list:
    mem = container.get_memory(user_id)
    return mem.get_all()


def clear_memory(user_id: str) -> None:
    mem = container.get_memory(user_id)
    mem.clear()

