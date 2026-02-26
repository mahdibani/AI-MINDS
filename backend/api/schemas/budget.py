from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class BudgetCheckRequest(BaseModel):
    item: str = Field(min_length=1)
    price: float = Field(gt=0)
    category: str = "discretionary"
    explain: bool = True
    budget_file: str
    safety_buffer_percent: float = Field(default=20, ge=0, le=60)
    user_id: str = "api_user"


class BudgetCheckResponse(BaseModel):
    affordable: bool
    confidence: float
    available_funds: float = 0
    monthly_income: float = 0
    total_expenses: float = 0
    discretionary_budget: float = 0
    current_savings: float = 0
    recommendation: str
    warnings: list[str] = []
    explanation: Optional[str] = ""


class BudgetSummaryRequest(BaseModel):
    budget_file: str
    safety_buffer_percent: float = Field(default=20, ge=0, le=60)
    user_id: str = "api_user"


class BudgetSummaryResponse(BaseModel):
    summary: str


class BudgetHistoryResponse(BaseModel):
    items: list[Any]

