from __future__ import annotations

from fastapi import APIRouter

from api.schemas.memory import (
    MemoryAddRequest,
    MemorySearchRequest,
    MemoryListRequest,
    MemoryAddResponse,
    MemoryItemsResponse,
    MemoryClearResponse,
)
from api.services.memory_service import add_memory, search_memory, list_memory, clear_memory

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/add", response_model=MemoryAddResponse)
def memory_add(payload: MemoryAddRequest) -> MemoryAddResponse:
    backend = add_memory(payload.user_id, payload.text, payload.metadata)
    return MemoryAddResponse(status="ok", backend=backend)


@router.post("/search", response_model=MemoryItemsResponse)
def memory_search(payload: MemorySearchRequest) -> MemoryItemsResponse:
    items = search_memory(payload.user_id, payload.query, payload.limit)
    return MemoryItemsResponse(items=items)


@router.post("/list", response_model=MemoryItemsResponse)
def memory_list(payload: MemoryListRequest) -> MemoryItemsResponse:
    items = list_memory(payload.user_id)
    return MemoryItemsResponse(items=items)


@router.post("/clear", response_model=MemoryClearResponse)
def memory_clear(payload: MemoryListRequest) -> MemoryClearResponse:
    clear_memory(payload.user_id)
    return MemoryClearResponse(status="cleared")

