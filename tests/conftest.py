import asyncio

import httpx
import pytest

from recommender import api as recommender_api
from recommender.api import app


def _startup_state() -> None:
    """Инициализирует состояние приложения без starlette.TestClient."""
    recommender_api.settings.n_candidates = 50
    recommender_api.init_db()
    recommender_api._load_services_from_registry()


class ASGISyncClient:
    """Минимальный sync-клиент поверх httpx.ASGITransport.

    В текущей связке starlette/httpx/anyio `fastapi.testclient.TestClient`
    зависает при входе в context manager, поэтому тесты используют ASGITransport
    и явную инициализацию состояния приложения.
    """

    def __enter__(self):
        _startup_state()
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def _call():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
                return await ac.request(method, url, **kwargs)

        return asyncio.run(_call())

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)


@pytest.fixture(scope="session")
def client():
    """HTTP-клиент с инициализированными моделями."""
    with ASGISyncClient() as c:
        yield c


@pytest.fixture
def client_factory():
    return ASGISyncClient


# ── Базовые тела запросов ────────────────────────────────────────────────────

@pytest.fixture
def recommend_body():
    return {
        "job_type": "pagerank",
        "input": {"profile": "large"},
        "topology": {"workers": 3, "worker_cores": 3, "worker_memory_gb": 6},
        "preferences": {"return_top_k": 3},
        "model_name": "rf",
    }


@pytest.fixture
def predict_body():
    return {
        "job_type": "pagerank",
        "profile": "large",
        "topology_workers": 3,
        "topology_worker_cores": 3,
        "topology_worker_mem_gb": 6,
        "executor_cores": 2,
        "executor_memory_mb": 3072,
        "executor_instances": 3,
    }
