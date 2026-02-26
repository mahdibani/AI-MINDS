from __future__ import annotations

from fastapi import FastAPI

from api.routers.health import router as health_router
from api.routers.repl import router as repl_router

app = FastAPI(title="AI-MINDS REPL Service", version="1.0.0")
app.include_router(health_router)
app.include_router(repl_router, prefix="/api/v1")

