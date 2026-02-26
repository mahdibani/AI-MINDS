from __future__ import annotations

from fastapi import APIRouter

from api.schemas.repl import ReplCodeRequest, ReplTaskRequest, ReplResponse
from api.services.repl_service import run_repl_code, run_repl_file_task

router = APIRouter(prefix="/repl", tags=["repl"])


@router.post("/code", response_model=ReplResponse)
def repl_code(payload: ReplCodeRequest) -> ReplResponse:
    try:
        result = run_repl_code(payload.code, payload.context, payload.max_iterations)
    except Exception as exc:
        result = f"Error: {exc}"
    return ReplResponse(result=result)


@router.post("/task", response_model=ReplResponse)
def repl_task(payload: ReplTaskRequest) -> ReplResponse:
    try:
        result = run_repl_file_task(payload.directory, payload.task, payload.max_iterations)
    except Exception as exc:
        result = f"Error: {exc}"
    return ReplResponse(result=result)
