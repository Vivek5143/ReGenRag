"""Versioned API router."""

from fastapi import APIRouter

from app.api.routes import documents, sessions

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(sessions.router)
api_router.include_router(documents.router)