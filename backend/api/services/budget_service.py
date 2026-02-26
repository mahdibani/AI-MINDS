from __future__ import annotations

from api.core.dependencies import container


def check_budget(
    budget_file: str,
    safety_buffer_percent: float,
    user_id: str,
    item: str,
    price: float,
    category: str,
    explain: bool,
) -> dict:
    advisor = container.get_budget_advisor(
        budget_file=budget_file.replace("\\", "/"),
        safety_buffer=safety_buffer_percent / 100.0,
        user_id=user_id,
    )
    return advisor.can_afford(item=item, price=price, category=category, explain=explain)


def get_budget_summary(budget_file: str, safety_buffer_percent: float, user_id: str) -> str:
    advisor = container.get_budget_advisor(
        budget_file=budget_file.replace("\\", "/"),
        safety_buffer=safety_buffer_percent / 100.0,
        user_id=user_id,
    )
    return advisor.get_budget_summary()


def get_budget_history(budget_file: str, safety_buffer_percent: float, user_id: str) -> list:
    advisor = container.get_budget_advisor(
        budget_file=budget_file.replace("\\", "/"),
        safety_buffer=safety_buffer_percent / 100.0,
        user_id=user_id,
    )
    return advisor.get_spending_history()

