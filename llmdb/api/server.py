"""FastAPI application factory and server entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from llmdb.engine import Database

# Module-level reference to the running Database instance
_db_instance: Database | None = None


def _app_db() -> Database:
    """Return the current Database instance (raises if server not started)."""
    if _db_instance is None:
        raise RuntimeError("Database not initialised. Call create_app() first.")
    return _db_instance


def create_app(db_path: str | Path = ":memory:") -> FastAPI:
    """Create and return the FastAPI application.

    Args:
        db_path: Path to the LLMDatabase file. Defaults to in-memory.
    """
    global _db_instance

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _db_instance
        _db_instance = Database(db_path)
        yield
        if _db_instance:
            _db_instance.close()
            _db_instance = None

    app = FastAPI(
        title="LLMDatabase",
        summary="A database management system reimagined for agentic applications.",
        version="0.1.0",
        lifespan=lifespan,
    )

    from llmdb.api.routes.collections import router as collections_router
    from llmdb.api.routes.documents import router as documents_router
    from llmdb.api.routes.memory import router as memory_router

    app.include_router(collections_router)
    app.include_router(documents_router)
    app.include_router(memory_router)

    @app.get("/", tags=["Health"])
    def root() -> dict[str, Any]:
        return {
            "service": "LLMDatabase",
            "version": "0.1.0",
            "status": "ok",
        }

    @app.get("/health", tags=["Health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def run(db_path: str | Path = ":memory:", host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the Uvicorn server programmatically."""
    import uvicorn

    app = create_app(db_path)
    uvicorn.run(app, host=host, port=port)
