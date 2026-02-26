from __future__ import annotations

from fastapi import APIRouter

from api.schemas.budget import (
    BudgetCheckRequest,
    BudgetCheckResponse,
    BudgetSummaryRequest,
    BudgetSummaryResponse,
    BudgetHistoryResponse,
)
from api.services.budget_service import check_budget, get_budget_history, get_budget_summary

router = APIRouter(prefix="/budget", tags=["budget"])


@router.post("/check", response_model=BudgetCheckResponse)
def budget_check(payload: BudgetCheckRequest) -> BudgetCheckResponse:
    result = check_budget(
        budget_file=payload.budget_file,
        safety_buffer_percent=payload.safety_buffer_percent,
        user_id=payload.user_id,
        item=payload.item,
        price=payload.price,
        category=payload.category,
        explain=payload.explain,
    )
    return BudgetCheckResponse(**result)


@router.post("/summary", response_model=BudgetSummaryResponse)
def budget_summary(payload: BudgetSummaryRequest) -> BudgetSummaryResponse:
    summary = get_budget_summary(
        budget_file=payload.budget_file,
        safety_buffer_percent=payload.safety_buffer_percent,
        user_id=payload.user_id,
    )
    return BudgetSummaryResponse(summary=summary)


@router.post("/history", response_model=BudgetHistoryResponse)
def budget_history(payload: BudgetSummaryRequest) -> BudgetHistoryResponse:
    items = get_budget_history(
        budget_file=payload.budget_file,
        safety_buffer_percent=payload.safety_buffer_percent,
        user_id=payload.user_id,
    )
    return BudgetHistoryResponse(items=items)

