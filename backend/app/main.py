import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.database import create_db_and_tables

# Import routers
from app.api import (
    routes_health,
    routes_projects,
    routes_brief,
    routes_concepts,
    routes_images,
    routes_feedback,
    routes_settings,
    routes_export
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-powered creative direction assistant for designers. One imagination. Ten creative directions.",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# Set CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database initialization on startup
@app.on_event("startup")
def on_startup():
    logger.info("Initializing database and tables...")
    create_db_and_tables()
    logger.info("Database initialized successfully.")

# Include routers
app.include_router(routes_health.router, prefix=settings.API_V1_STR)
app.include_router(routes_projects.router, prefix=settings.API_V1_STR)
app.include_router(routes_brief.router, prefix=settings.API_V1_STR)
app.include_router(routes_concepts.router, prefix=settings.API_V1_STR)
app.include_router(routes_images.router, prefix=settings.API_V1_STR)
app.include_router(routes_feedback.router, prefix=settings.API_V1_STR)
app.include_router(routes_settings.router, prefix=settings.API_V1_STR)
app.include_router(routes_export.router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to SIFS Imagination Expander AI API Backend.",
        "documentation": "/docs"
    }
