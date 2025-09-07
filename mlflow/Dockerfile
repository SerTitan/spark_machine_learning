FROM python:3.11-slim

# 1) Установим MLflow (можно поменять версию при желании)
RUN pip install --no-cache-dir mlflow==2.15.1

# 2) Каталоги для storage и артефактов
VOLUME ["/mlflow/mlruns", "/mlflow/mlartifacts"]
EXPOSE 5000

# 3) Запуск MLflow (shell-форма, одна строка)
CMD mlflow server --backend-store-uri file:/mlflow/mlruns --default-artifact-root file:/mlflow/mlartifacts --host 0.0.0.0 --port 5000
