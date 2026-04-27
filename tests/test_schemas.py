"""Unit-тесты Pydantic-схем через FastAPI 422-ответы."""

import pytest


class TestRecommendRequest:
    def test_valid_minimal(self, client):
        r = client.post("/recommend", json={
            "job_type": "pagerank",
            "input": {"profile": "large"},
            "topology": {"workers": 3, "worker_cores": 3, "worker_memory_gb": 6},
        })
        assert r.status_code == 200

    def test_missing_job_type(self, client):
        r = client.post("/recommend", json={
            "input": {"profile": "large"},
            "topology": {"workers": 3, "worker_cores": 3, "worker_memory_gb": 6},
        })
        assert r.status_code == 422
        body = r.json()
        assert any("job_type" in str(e["loc"]) for e in body["detail"])

    def test_missing_topology(self, client):
        r = client.post("/recommend", json={
            "job_type": "pagerank",
            "input": {"profile": "large"},
        })
        assert r.status_code == 422

    def test_missing_profile(self, client):
        r = client.post("/recommend", json={
            "job_type": "pagerank",
            "input": {},
            "topology": {"workers": 3, "worker_cores": 3, "worker_memory_gb": 6},
        })
        assert r.status_code == 422

    def test_topology_workers_zero(self, client):
        r = client.post("/recommend", json={
            "job_type": "pagerank",
            "input": {"profile": "large"},
            "topology": {"workers": 0, "worker_cores": 3, "worker_memory_gb": 6},
        })
        assert r.status_code == 422

    def test_topology_workers_negative(self, client):
        r = client.post("/recommend", json={
            "job_type": "pagerank",
            "input": {"profile": "large"},
            "topology": {"workers": -1, "worker_cores": 3, "worker_memory_gb": 6},
        })
        assert r.status_code == 422

    def test_top_k_zero(self, client):
        r = client.post("/recommend", json={
            "job_type": "pagerank",
            "input": {"profile": "large"},
            "topology": {"workers": 3, "worker_cores": 3, "worker_memory_gb": 6},
            "preferences": {"return_top_k": 0},
        })
        assert r.status_code == 422

    def test_top_k_above_max(self, client):
        r = client.post("/recommend", json={
            "job_type": "pagerank",
            "input": {"profile": "large"},
            "topology": {"workers": 3, "worker_cores": 3, "worker_memory_gb": 6},
            "preferences": {"return_top_k": 100},
        })
        assert r.status_code == 422

    def test_unsupported_job_type_returns_422(self, client):
        r = client.post("/recommend", json={
            "job_type": "nonexistent_workload",
            "input": {"profile": "large"},
            "topology": {"workers": 3, "worker_cores": 3, "worker_memory_gb": 6},
        })
        assert r.status_code == 422
        assert "not supported" in r.json()["detail"]


class TestPredictRequest:
    def test_valid_minimal(self, client):
        r = client.post("/predict", json={
            "job_type": "pagerank",
            "profile": "large",
            "topology_workers": 3,
            "topology_worker_cores": 3,
            "topology_worker_mem_gb": 6,
            "executor_cores": 2,
            "executor_memory_mb": 3072,
            "executor_instances": 3,
        })
        assert r.status_code == 200

    def test_missing_executor_cores(self, client):
        r = client.post("/predict", json={
            "job_type": "pagerank",
            "profile": "large",
            "topology_workers": 3,
            "topology_worker_cores": 3,
            "topology_worker_mem_gb": 6,
            "executor_memory_mb": 3072,
            "executor_instances": 3,
        })
        assert r.status_code == 422

    def test_executor_memory_below_min(self, client):
        r = client.post("/predict", json={
            "job_type": "pagerank",
            "profile": "large",
            "topology_workers": 3,
            "topology_worker_cores": 3,
            "topology_worker_mem_gb": 6,
            "executor_cores": 2,
            "executor_memory_mb": 100,  # ниже минимума 256
            "executor_instances": 3,
        })
        assert r.status_code == 422
