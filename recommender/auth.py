"""
Двухуровневая аутентификация X-API-Key.

Роли:
  user  — все публичные эндпоинты без ключа:
           /health, /jobs, /metrics, /predict, /recommend, /history
  admin — дополнительно /admin/* (управление моделями), требует ключ

Публичные эндпоинты могут работать без ключа. Админ-эндпоинты всегда требуют
конкретный ключ: RECOMMENDER_ADMIN_KEY или fallback RECOMMENDER_API_KEY.
"""

from fastapi import Header, HTTPException
from .settings import settings


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> str | None:
    """Публичные эндпоинты: ключ не обязателен. Неверный ключ → 401."""
    if not settings.api_key:
        return None
    if x_api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")
    return f"...{x_api_key[-4:]}" if (x_api_key and len(x_api_key) >= 4) else None


async def require_admin(x_api_key: str | None = Header(default=None)) -> str:
    """Административные эндпоинты: ключ обязателен всегда."""
    expected = settings.admin_key or settings.api_key
    if not expected:
        raise HTTPException(
            status_code=403,
            detail="Admin password is not configured. Set RECOMMENDER_ADMIN_KEY.",
        )
    if x_api_key != expected:
        raise HTTPException(
            status_code=403,
            detail="Admin access required. Provide X-API-Key header.",
        )
    return f"...{x_api_key[-4:]}" if len(x_api_key) >= 4 else "****"
