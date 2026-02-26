from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class MemoryAddRequest(BaseModel):
    user_id: str = "api_user"
    text: str = Field(min_length=1)
    metadata: Optional[dict[str, Any]] = None


class MemorySearchRequest(BaseModel):
    user_id: str = "api_user"
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


class MemoryListRequest(BaseModel):
    user_id: str = "api_user"


class MemoryAddResponse(BaseModel):
    status: str
    backend: str


class MemoryItemsResponse(BaseModel):
    items: list[Any]


class MemoryClearResponse(BaseModel):
    status: str

