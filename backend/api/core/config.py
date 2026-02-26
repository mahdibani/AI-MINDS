from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "AI-MINDS API Gateway"
    app_version: str = "1.0.0"
    host: str = Field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    default_budget_path: str = Field(
        default_factory=lambda: os.getenv(
            "BUDGET_FILE_DEFAULT",
            str(Path.home() / "Downloads" / "budget.xlsx"),
        )
    )
    default_search_dir: str = Field(
        default_factory=lambda: os.getenv(
            "DEFAULT_SEARCH_DIR",
            str(Path.home() / "Downloads"),
        )
    )
    default_user_id: str = Field(default_factory=lambda: os.getenv("DEFAULT_USER_ID", "api_user"))


settings = Settings()

