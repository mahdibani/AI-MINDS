from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ListFilesRequest(BaseModel):
    directory: str


class ParseFileRequest(BaseModel):
    filepath: str
    max_chars: int = Field(default=50000, ge=1000, le=200000)


class FileAnalysisRequest(BaseModel):
    directory: str
    query: str = Field(min_length=1)
    max_iterations: int = Field(default=8, ge=2, le=25)


class FSResponse(BaseModel):
    data: dict[str, Any]


class FileAnalysisResponse(BaseModel):
    result: str

