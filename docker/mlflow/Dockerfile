FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.15.1

VOLUME ["/mlflow/mlruns", "/mlflow/mlartifacts"]
EXPOSE 5000

CMD mlflow server --backend-store-uri file:/mlflow/mlruns --default-artifact-root file:/mlflow/mlartifacts --host 0.0.0.0 --port 5000
