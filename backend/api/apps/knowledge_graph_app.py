from __future__ import annotations

from fastapi import FastAPI

from api.routers.health import router as health_router
from api.routers.knowledge_graph import router as knowledge_graph_router

app = FastAPI(title="AI-MINDS Knowledge Graph Service", version="1.0.0")
app.include_router(health_router)
app.include_router(knowledge_graph_router, prefix="/api/v1")

