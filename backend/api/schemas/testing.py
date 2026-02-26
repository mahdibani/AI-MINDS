from __future__ import annotations

from pydantic import BaseModel


class TestRunRequest(BaseModel):
    verbose: bool = False


class TestRunResponse(BaseModel):
    command: str
    return_code: int
    output: str

