"""Unit-тесты аутентификации X-API-Key."""

import pytest
from unittest.mock import patch


@pytest.fixture
def authed_client(client_factory):
    """Клиент с включённой аутентификацией через env."""
    with patch("recommender.auth.settings") as mock_auth_settings:
        mock_auth_settings.api_key = "secret-test-key"
        mock_auth_settings.admin_key = "secret-test-key"
        with client_factory() as c:
            yield c


class TestAuthDisabled:
    def test_no_key_allowed_when_auth_disabled(self, client):
        r = client.post("/recommend", json={
            "job_type": "pagerank",
            "input": {"profile": "large"},
            "topology": {"workers": 2, "worker_cores": 4, "worker_memory_gb": 8},
        })
        assert r.status_code == 200

    def test_any_key_allowed_when_auth_disabled(self, client):
        r = client.post("/predict", json={
            "job_type": "pagerank", "profile": "large",
            "topology_workers": 2, "topology_worker_cores": 4, "topology_worker_mem_gb": 8,
            "executor_cores": 1, "executor_memory_mb": 1024, "executor_instances": 1,
        }, headers={"X-API-Key": "whatever"})
        assert r.status_code == 200

    def test_admin_endpoint_is_closed_without_configured_key(self, client):
        r = client.get("/admin/models", headers={"X-API-Key": "whatever"})
        assert r.status_code == 403


class TestAuthEnabled:
    def test_valid_key_allowed(self, authed_client):
        r = authed_client.post("/recommend", json={
            "job_type": "pagerank",
            "input": {"profile": "large"},
            "topology": {"workers": 2, "worker_cores": 4, "worker_memory_gb": 8},
        }, headers={"X-API-Key": "secret-test-key"})
        assert r.status_code == 200

    def test_wrong_key_rejected(self, authed_client):
        r = authed_client.post("/recommend", json={
            "job_type": "pagerank",
            "input": {"profile": "large"},
            "topology": {"workers": 2, "worker_cores": 4, "worker_memory_gb": 8},
        }, headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401

    def test_public_endpoint_accessible_without_key(self, authed_client):
        """Публичные эндпоинты доступны без ключа даже при включённой аутентификации."""
        r = authed_client.post("/recommend", json={
            "job_type": "pagerank",
            "input": {"profile": "large"},
            "topology": {"workers": 2, "worker_cores": 4, "worker_memory_gb": 8},
        })
        assert r.status_code == 200

    def test_admin_endpoint_requires_key(self, authed_client):
        """Администраторские эндпоинты возвращают 403 без ключа."""
        r = authed_client.get("/admin/models")
        assert r.status_code == 403

    def test_admin_endpoint_valid_key(self, authed_client):
        """Администраторские эндпоинты доступны с правильным ключом."""
        r = authed_client.get("/admin/models", headers={"X-API-Key": "secret-test-key"})
        assert r.status_code == 200

    def test_admin_endpoint_rejects_wrong_key(self, authed_client):
        r = authed_client.get("/admin/models", headers={"X-API-Key": "any-string"})
        assert r.status_code == 403
