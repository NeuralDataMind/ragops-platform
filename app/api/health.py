from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.db.postgres import check_postgres
from app.db.qdrant import check_qdrant
from app.db.redis import check_redis

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "enterprise-rag-platform"
    }

@router.get("/ready")
async def readiness_check():
    checks = {
        "postgres": await check_postgres(),
        "qdrant": await check_qdrant(),
        "redis": await check_redis(),
    }

    is_ready = all(checks.values())

    response = {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
    }

    if not is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response,
        )

    return response