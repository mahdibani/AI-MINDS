from __future__ import annotations

from fastapi import APIRouter

from api.schemas.knowledge_graph import (
    KnowledgeGraphBuildRequest,
    KnowledgeGraphBuildResponse,
    KnowledgeGraphSearchRequest,
    KnowledgeGraphSearchResponse,
)
from api.services.knowledge_graph_service import build_knowledge_graph, search_knowledge_graph

router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])


@router.post("/build", response_model=KnowledgeGraphBuildResponse)
async def knowledge_graph_build(payload: KnowledgeGraphBuildRequest) -> KnowledgeGraphBuildResponse:
    try:
        data = await build_knowledge_graph(payload.directory, payload.mode)
        return KnowledgeGraphBuildResponse(**data)
    except Exception as exc:
        return KnowledgeGraphBuildResponse(
            graph_id="",
            summary=f"Error: {exc}",
            html_path="",
            stats={},
        )


@router.post("/search", response_model=KnowledgeGraphSearchResponse)
async def knowledge_graph_search(payload: KnowledgeGraphSearchRequest) -> KnowledgeGraphSearchResponse:
    results = await search_knowledge_graph(payload.graph_id, payload.query, payload.limit)
    return KnowledgeGraphSearchResponse(results=results)
