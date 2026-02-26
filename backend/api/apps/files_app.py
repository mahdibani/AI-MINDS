from __future__ import annotations

from fastapi import FastAPI

from api.routers.health import router as health_router
from api.routers.files import router as files_router

app = FastAPI(title="AI-MINDS Files Service", version="1.0.0")
app.include_router(health_router)
app.include_router(files_router, prefix="/api/v1")

