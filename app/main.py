from fastapi import FastAPI

from contextlib import asynccontextmanager

from app.api.health import router as heath_router
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router
from app.core.config import settings
from app.db.postgres import connect_postgres, close_postgres
from app.db.schema import init_db_schema

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_postgres()
    await init_db_schema()
    yield
    await close_postgres()

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    debug=settings.DEBUG,
    lifespan=lifespan
)

app.include_router(heath_router)
app.include_router(documents_router, prefix="/documents", tags=["Documents"])
app.include_router(chat_router, prefix="/chat", tags=["Chat"])