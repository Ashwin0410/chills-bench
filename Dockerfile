FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bench.py bench.html ./

ENV BENCH_DATA_DIR=/data/bench

EXPOSE 8000

CMD ["sh", "-c", "uvicorn bench:app --host 0.0.0.0 --port ${PORT:-8000}"]