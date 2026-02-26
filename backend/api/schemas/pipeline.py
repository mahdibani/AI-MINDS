from __future__ import annotations

from pydantic import BaseModel, Field


class PipelineRunRequest(BaseModel):
    prompt: str = Field(min_length=1)
    task: str = Field(min_length=1)
    chunk_size: int = Field(default=80000, ge=10000, le=250000)
    max_workers: int = Field(default=4, ge=1, le=16)
    parallel: bool = True


class PipelineRunResponse(BaseModel):
    final_output: str
    chunks_processed: int
    total_tokens: int
    total_cost: float
    elapsed_total: float

