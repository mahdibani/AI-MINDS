from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import settings
from api.routers.health import router as health_router
from api.routers.budget import router as budget_router
from api.routers.files import router as files_router
from api.routers.memory import router as memory_router
from api.routers.pipeline import router as pipeline_router
from api.routers.knowledge_graph import router as knowledge_graph_router
from api.routers.repl import router as repl_router
from api.routers.testing import router as testing_router


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(budget_router, prefix="/api/v1")
    app.include_router(files_router, prefix="/api/v1")
    app.include_router(memory_router, prefix="/api/v1")
    app.include_router(pipeline_router, prefix="/api/v1")
    app.include_router(knowledge_graph_router, prefix="/api/v1")
    app.include_router(repl_router, prefix="/api/v1")
    app.include_router(testing_router, prefix="/api/v1")
    return app


app = create_app()
