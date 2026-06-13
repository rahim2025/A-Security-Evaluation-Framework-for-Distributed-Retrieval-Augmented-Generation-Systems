FROM python:3.10-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY attack/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY attack /app/attack
COPY data   /app/data

# Default URLs — overridden in docker-compose when running beside other containers
ENV DATA_SOURCE_0_URL=http://host.docker.internal:8001
ENV DATA_SOURCE_20_URL=http://host.docker.internal:8002
ENV DATA_SOURCE_100_URL=http://host.docker.internal:8003
ENV LLM_SERVICE_URL=http://host.docker.internal:9000
ENV DATA_DIR=/app/data/polluted_token

ENTRYPOINT ["python", "attack/datapoisoning/run_attack.py"]
