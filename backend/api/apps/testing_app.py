from __future__ import annotations

from fastapi import FastAPI

from api.routers.health import router as health_router
from api.routers.testing import router as testing_router

app = FastAPI(title="AI-MINDS Testing Service", version="1.0.0")
app.include_router(health_router)
app.include_router(testing_router, prefix="/api/v1")

