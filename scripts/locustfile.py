"""
Нагрузочный тест рекомендательного сервиса.

Запуск (headless, 30 сек, 10 пользователей):
  .venv/bin/locust -f scripts/locustfile.py \
    --headless -u 10 -r 2 -t 30s \
    --host http://localhost:8001 \
    --html out/locust_report.html

Запуск с Web UI (открыть http://localhost:8089):
  .venv/bin/locust -f scripts/locustfile.py --host http://localhost:8001
"""

import random
from locust import HttpUser, between, task

_TOPOLOGIES = [
    {"workers": 4, "worker_cores": 2, "worker_memory_gb": 4},
    {"workers": 2, "worker_cores": 4, "worker_memory_gb": 8},
    {"workers": 3, "worker_cores": 3, "worker_memory_gb": 6},
]

_PREDICT_BODY = {
    "job_type": "pagerank", "profile": "large",
    "topology_workers": 4, "topology_worker_cores": 2, "topology_worker_mem_gb": 4,
    "executor_cores": 2, "executor_memory_mb": 2048, "executor_instances": 2,
    "driver_cores": 1, "driver_memory_mb": 1024,
    "memory_fraction": 0.6, "memory_storageFraction": 0.5,
    "shuffle_compress": 1, "spill_compress": 1,
    "shuffle_file_buffer_kb": 32, "broadcast_block_mb": 4,
    "broadcast_compress": 1, "maxSizeInFlight_mb": 48,
    "rpc_message_maxSize": 128, "rdd_compress": 0, "io_codec": "lz4",
}


class RecommenderUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(3)
    def recommend(self):
        topo = random.choice(_TOPOLOGIES)
        profile = random.choice(["small", "large"])
        top_k = random.randint(1, 3)
        self.client.post("/recommend", json={
            "job_type": "pagerank",
            "input": {"profile": profile},
            "topology": topo,
            "preferences": {"return_top_k": top_k},
        }, name="/recommend")

    @task(2)
    def predict(self):
        self.client.post("/predict", json=_PREDICT_BODY, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")

    @task(1)
    def history(self):
        limit = random.choice([10, 25, 50])
        self.client.get(f"/history?limit={limit}", name="/history")

    @task(1)
    def metrics(self):
        self.client.get("/metrics", name="/metrics")
