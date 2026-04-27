FROM python:3.12-slim AS deps
WORKDIR /deps
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

FROM python:3.12-slim
WORKDIR /app

COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY recommender/ ./recommender/

EXPOSE 8000

ENV RECOMMENDER_MODEL_DIR=/app/out/final_best

CMD ["uvicorn", "recommender.api:app", "--host", "0.0.0.0", "--port", "8000"]
