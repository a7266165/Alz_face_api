"""
src/api/routers
API 路由模組
"""

from fastapi import APIRouter
from . import analyze, health

# 建立主路由器
api_router = APIRouter()

# 註冊子路由
api_router.include_router(analyze.router)
api_router.include_router(health.router)

__all__ = [
    "api_router",
    "analyze",
    "health",
]