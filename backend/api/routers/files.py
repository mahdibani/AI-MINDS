from __future__ import annotations

from fastapi import APIRouter

from api.schemas.files import (
    ListFilesRequest,
    ParseFileRequest,
    FileAnalysisRequest,
    FSResponse,
    FileAnalysisResponse,
)
from api.services.files_service import list_files, parse_file, analyze_files

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/list", response_model=FSResponse)
def files_list(payload: ListFilesRequest) -> FSResponse:
    return FSResponse(data=list_files(payload.directory))


@router.post("/parse", response_model=FSResponse)
def files_parse(payload: ParseFileRequest) -> FSResponse:
    return FSResponse(data=parse_file(payload.filepath, payload.max_chars))


@router.post("/analyze", response_model=FileAnalysisResponse)
def files_analyze(payload: FileAnalysisRequest) -> FileAnalysisResponse:
    result = analyze_files(payload.directory, payload.query, payload.max_iterations)
    return FileAnalysisResponse(result=result)

