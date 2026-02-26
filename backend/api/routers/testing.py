from __future__ import annotations

from fastapi import APIRouter

from api.schemas.testing import TestRunRequest, TestRunResponse
from api.services.testing_service import run_budget_tests, run_rlm_tests

router = APIRouter(prefix="/testing", tags=["testing"])


@router.post("/budget", response_model=TestRunResponse)
def testing_budget(payload: TestRunRequest) -> TestRunResponse:
    data = run_budget_tests(payload.verbose)
    return TestRunResponse(**data)


@router.post("/rlm", response_model=TestRunResponse)
def testing_rlm(payload: TestRunRequest) -> TestRunResponse:
    data = run_rlm_tests(payload.verbose)
    return TestRunResponse(**data)

