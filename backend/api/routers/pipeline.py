from __future__ import annotations

from fastapi import APIRouter

from api.schemas.pipeline import PipelineRunRequest, PipelineRunResponse
from api.services.pipeline_service import run_pipeline

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/run", response_model=PipelineRunResponse)
def pipeline_run(payload: PipelineRunRequest) -> PipelineRunResponse:
    data = run_pipeline(
        prompt=payload.prompt,
        task=payload.task,
        chunk_size=payload.chunk_size,
        max_workers=payload.max_workers,
        parallel=payload.parallel,
    )
    return PipelineRunResponse(**data)

