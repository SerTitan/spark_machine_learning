# Spark Benchmark Cluster (Spark 4.0.0)

## Сервисы
- Apache Spark 4.0.0 (Master + Worker)
- Spark History Server
- Prometheus 2.52.0
- MLflow Tracking Server 2.12.1

## Запуск
```bash
docker-compose build
docker-compose up -d
```

## Интерфейсы
- Spark Master UI: http://localhost:8080
- History Server:  http://localhost:18080
- Prometheus UI:   http://localhost:9090
- MLflow UI:       http://localhost:5000
```
