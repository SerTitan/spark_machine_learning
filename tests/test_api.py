"""Integration-тесты всех эндпоинтов REST API."""

import base64
import shutil

import pytest

from recommender import api as recommender_api
from recommender.model_registry import ModelRegistry
from recommender.settings import settings


class TestHealth:
    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_models_loaded(self, client):
        r = client.get("/health")
        assert len(r.json()["models_loaded"]) > 0

    def test_model_dir_present(self, client):
        r = client.get("/health")
        assert "model_dir" in r.json()

    def test_skips_dnn_when_torch_missing(self, monkeypatch, client_factory):
        class MissingTorchDnn:
            def __init__(self, artifact):
                raise ModuleNotFoundError("No module named 'torch'", name="torch")

        monkeypatch.setattr(recommender_api, "DnnPredictorService", MissingTorchDnn)
        try:
            with client_factory() as c:
                body = c.get("/health").json()
                assert "pagerank/rf" in body["models_loaded"]
                assert "pagerank/dnn" not in body["models_loaded"]
                assert any("PyTorch is not installed" in w for w in body["model_warnings"])
        finally:
            monkeypatch.undo()
            recommender_api._load_services_from_registry()


class TestJobs:
    def test_returns_200(self, client):
        assert client.get("/jobs").status_code == 200

    def test_supported_contains_pagerank(self, client):
        supported = client.get("/jobs").json()["supported"]
        assert "pagerank" in supported

    def test_models_loaded_nonempty(self, client):
        assert len(client.get("/jobs").json()["models_loaded"]) > 0

    def test_models_loaded_contains_job_type_and_model(self, client):
        items = client.get("/jobs").json()["models_loaded"]
        assert any(item["job_type"] == "pagerank" for item in items)


class TestMetrics:
    def test_returns_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_contains_mae_and_r2(self, client):
        models = client.get("/metrics").json()["models"]
        for jt, info in models.items():
            if jt.endswith("/ql"):
                continue
            m = info["metrics"]
            assert "MAE" in m and "R2" in m, f"{jt} missing MAE or R2"

    def test_dataset_version_present(self, client):
        models = client.get("/metrics").json()["models"]
        for jt, info in models.items():
            assert info["dataset_version"], f"{jt} missing dataset_version"

    def test_pagerank_rf_metrics_match_loaded_model_not_dummy(self, client):
        model = client.get("/metrics").json()["models"]["pagerank/rf"]
        metrics = model["metrics"]
        assert model["predictor_type"] == "randomforest_randomsearch"
        assert metrics["MAE"] < 20
        assert metrics["R2"] > 0

    def test_all_recommendation_strategies_loaded(self, client):
        models = client.get("/metrics").json()["models"]
        assert {"pagerank/rf", "pagerank/dnn", "pagerank/ql"}.issubset(models)


class TestPredict:
    def test_happy_path(self, client, predict_body):
        r = client.post("/predict", json=predict_body)
        assert r.status_code == 200
        body = r.json()
        assert body["predicted_runtime_s"] > 0
        assert len(body["confidence_band"]) == 2
        assert body["confidence_band"][0] <= body["confidence_band"][1]

    def test_executor_cores_warning(self, client, predict_body):
        body = dict(predict_body, executor_cores=99)  # заведомо > worker_cores
        r = client.post("/predict", json=body)
        assert r.status_code == 200
        assert any("executor_cores" in w for w in r.json()["warnings"])

    def test_executor_memory_warning(self, client, predict_body):
        body = dict(predict_body, executor_memory_mb=999_999)
        r = client.post("/predict", json=body)
        assert r.status_code == 200
        assert any("executor_memory_mb" in w for w in r.json()["warnings"])

    def test_model_type_in_response(self, client, predict_body):
        r = client.post("/predict", json=predict_body)
        assert r.json()["model_type"]

    def test_predict_supports_dnn_and_qlearning_strategies(self, client, predict_body):
        for model_name in ("dnn", "ql"):
            r = client.post("/predict", json={**predict_body, "model_name": model_name})
            assert r.status_code == 200
            body = r.json()
            assert body["predicted_runtime_s"] > 0
            assert body["model_info"]["predictor_type"] == model_name

    def test_non_grid_numeric_value_warns(self, client, predict_body):
        body = dict(predict_body, executor_memory_mb=1536)
        r = client.post("/predict", json=body)
        assert r.status_code == 200
        assert any("executor_memory_mb" in w and "outside sampled design grid" in w
                   for w in r.json()["warnings"])

    def test_unknown_codec_warns(self, client, predict_body):
        body = dict(predict_body, io_codec="zstd")
        r = client.post("/predict", json=body)
        assert r.status_code == 200
        assert any("io_codec" in w and "not present" in w for w in r.json()["warnings"])


class TestRecommend:
    def test_happy_path(self, client, recommend_body):
        r = client.post("/recommend", json=recommend_body)
        assert r.status_code == 200
        recs = r.json()["recommendations"]
        assert len(recs) == 3

    def test_ranks_sequential(self, client, recommend_body):
        recs = client.post("/recommend", json=recommend_body).json()["recommendations"]
        assert [r["rank"] for r in recs] == list(range(1, len(recs) + 1))

    def test_sorted_by_runtime(self, client, recommend_body):
        recs = client.post("/recommend", json=recommend_body).json()["recommendations"]
        times = [r["predicted_runtime_s"] for r in recs]
        assert times == sorted(times)

    def test_speedup_positive(self, client, recommend_body):
        recs = client.post("/recommend", json=recommend_body).json()["recommendations"]
        assert all(r["predicted_speedup_vs_default"] > 0 for r in recs)

    def test_confidence_band_ordered(self, client, recommend_body):
        recs = client.post("/recommend", json=recommend_body).json()["recommendations"]
        for r in recs:
            lo, hi = r["confidence_band"]
            assert lo <= hi + 0.01

    def test_top_k_respected(self, client, recommend_body):
        for k in (1, 2, 5):
            body = dict(recommend_body, preferences={"return_top_k": k})
            recs = client.post("/recommend", json=body).json()["recommendations"]
            assert len(recs) <= k

    def test_no_duplicate_executor_configs(self, client, recommend_body):
        body = dict(recommend_body, preferences={"return_top_k": 5})
        recs = client.post("/recommend", json=body).json()["recommendations"]
        keys = [(r["config"]["executor_cores"],
                 r["config"]["executor_memory_mb"],
                 r["config"]["executor_instances"]) for r in recs]
        assert len(keys) == len(set(keys))

    def test_model_info_in_response(self, client, recommend_body):
        mi = client.post("/recommend", json=recommend_body).json()["model_info"]
        assert mi["job_type"] == "pagerank"
        assert mi["predictor_mae_s"] is not None
        assert mi["predictor_r2"] is not None

    def test_repeated_requests_return_valid_recommendations(self, client, recommend_body):
        for _ in range(2):
            recs = client.post("/recommend", json=recommend_body).json()["recommendations"]
            assert len(recs) == recommend_body["preferences"]["return_top_k"]
            for r in recs:
                assert r["predicted_runtime_s"] > 0
                assert r["rank"] >= 1

    def test_constraints_max_executor_cores(self, client, recommend_body):
        limit = 2
        body = dict(recommend_body, constraints={"max_executor_cores": limit})
        recs = client.post("/recommend", json=body).json()["recommendations"]
        assert all(r["config"]["executor_cores"] <= limit for r in recs)

    def test_constraints_max_executor_memory(self, client, recommend_body):
        limit_mb = 2048
        body = dict(recommend_body, constraints={"max_executor_memory_mb": limit_mb})
        recs = client.post("/recommend", json=body).json()["recommendations"]
        assert all(r["config"]["executor_memory_mb"] <= limit_mb for r in recs)

    def test_constraints_max_instances(self, client, recommend_body):
        limit = 1
        body = dict(recommend_body, constraints={"max_executor_instances": limit})
        recs = client.post("/recommend", json=body).json()["recommendations"]
        assert all(r["config"]["executor_instances"] <= limit for r in recs)

    def test_dnn_strategy_works_when_loaded(self, client, recommend_body):
        body = dict(recommend_body, model_name="dnn", preferences={"return_top_k": 1})
        r = client.post("/recommend", json=body)
        assert r.status_code == 200
        assert r.json()["model_info"]["predictor_type"] == "dnn"
        assert len(r.json()["recommendations"]) == 1

    def test_qlearning_strategy_works_when_loaded(self, client, recommend_body):
        body = dict(recommend_body, model_name="ql", preferences={"return_top_k": 3})
        r = client.post("/recommend", json=body)
        assert r.status_code == 200
        payload = r.json()
        assert payload["model_info"]["predictor_type"] == "ql"
        assert payload["model_info"]["predictor_mae_s"] is not None
        assert payload["model_info"]["predictor_r2"] is not None
        assert len(payload["recommendations"]) == 3


class TestHistory:
    def test_returns_list(self, client):
        r = client.get("/history")
        assert r.status_code == 200
        assert "items" in r.json()

    def test_stats_endpoint(self, client):
        r = client.get("/history/stats")
        assert r.status_code == 200
        body = r.json()
        assert "total_requests" in body
        assert "by_endpoint" in body
        assert "avg_duration_ms" in body

    def test_predict_logged(self, client, predict_body):
        client.post("/predict", json=predict_body)
        items = client.get("/history?limit=1").json()["items"]
        assert items[0]["endpoint"] == "/predict"
        assert isinstance(items[0]["ts_ms"], int)
        assert items[0]["ts_ms"] > items[0]["ts"]
        assert items[0]["request_json"]["executor_cores"] == predict_body["executor_cores"]
        assert "predicted_runtime_s" in items[0]["response_json"]

    def test_recommend_logged(self, client, recommend_body):
        client.post("/recommend", json=recommend_body)
        items = client.get("/history?limit=1").json()["items"]
        assert items[0]["endpoint"] == "/recommend"
        assert items[0]["request_json"]["topology"] == recommend_body["topology"]
        assert "recommendations" in items[0]["response_json"]


class TestUI:
    def test_root_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Spark Config Recommender" in r.text
        assert "Прогноз времени" in r.text
        assert 'data-panel="recommend predict"' in r.text


class TestAdminModelManagement:
    @pytest.fixture
    def temp_model_client(self, tmp_path, monkeypatch, client_factory):
        src = settings.model_dir / "pagerank"
        dst_root = tmp_path / "models"
        shutil.copytree(src, dst_root / "pagerank")

        monkeypatch.setattr(recommender_api.settings, "model_dir", dst_root)
        monkeypatch.setattr(recommender_api, "registry", ModelRegistry(dst_root))
        monkeypatch.setattr("recommender.auth.settings.admin_key", "secret-test-key")
        monkeypatch.setattr("recommender.auth.settings.api_key", None)

        with client_factory() as c:
            yield c, dst_root

    def test_delete_model_removes_it_from_pool(self, temp_model_client, recommend_body):
        client, _ = temp_model_client
        r = client.delete("/admin/models/pagerank/rf", headers={"X-API-Key": "secret-test-key"})
        assert r.status_code == 200
        assert "pagerank/rf" not in client.get("/health").json()["models_loaded"]

        rec = client.post("/recommend", json=recommend_body)
        assert rec.status_code == 422

    def test_upload_model_adds_it_back_to_pool(self, temp_model_client, recommend_body):
        client, root = temp_model_client
        model_bytes = (root / "pagerank" / "model_randomforest_randomsearch.joblib").read_bytes()
        pre_bytes = (root / "pagerank" / "preprocessor.joblib").read_bytes()

        client.delete("/admin/models/pagerank/rf", headers={"X-API-Key": "secret-test-key"})
        body = {
            "job_type": "pagerank",
            "model_name": "rf",
            "filename": "model.joblib",
            "content_b64": base64.b64encode(model_bytes).decode("ascii"),
            "preprocessor_filename": "preprocessor.joblib",
            "preprocessor_content_b64": base64.b64encode(pre_bytes).decode("ascii"),
        }
        r = client.post("/admin/models/upload", json=body, headers={"X-API-Key": "secret-test-key"})
        assert r.status_code == 200
        assert "pagerank/rf" in client.get("/health").json()["models_loaded"]
        assert client.post("/recommend", json=recommend_body).status_code == 200
