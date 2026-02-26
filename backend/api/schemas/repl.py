from __future__ import annotations

from pydantic import BaseModel, Field


class ReplCodeRequest(BaseModel):
    code: str = Field(min_length=1)
    context: str = "REPL execution"
    max_iterations: int = Field(default=5, ge=2, le=20)


class ReplTaskRequest(BaseModel):
    directory: str
    task: str = Field(min_length=1)
    max_iterations: int = Field(default=10, ge=2, le=30)


class ReplResponse(BaseModel):
    result: str

