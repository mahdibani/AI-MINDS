from __future__ import annotations

from pydantic import BaseModel, Field


class KnowledgeGraphBuildRequest(BaseModel):
    directory: str
    mode: str = Field(default="auto", pattern="^(auto|cognee|networkx)$")


class KnowledgeGraphBuildResponse(BaseModel):
    graph_id: str
    summary: str
    html_path: str
    stats: dict


class KnowledgeGraphSearchRequest(BaseModel):
    graph_id: str
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=50)


class KnowledgeGraphSearchResponse(BaseModel):
    results: list[str]

