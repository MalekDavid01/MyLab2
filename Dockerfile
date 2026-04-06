# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# git is required by pip to install segment-anything from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy inference server only (training/eval scripts are not needed at runtime)
COPY app.py .

# Checkpoint directory — model weights are NOT baked into the image.
# Mount your trained checkpoint at runtime:
#   docker run -v $(pwd)/checkpoints:/app/checkpoints ...
RUN mkdir -p /app/checkpoints/best

# ── Runtime config (override with --env or --env-file at docker run time) ─────
ENV PORT=5000 \
    IMG_SIZE=512 \
    CHECKPOINT_DIR=checkpoints/best \
    PYTHONUNBUFFERED=1

# Secrets such as HF_TOKEN must NEVER be baked into the image.
# Pass them at runtime: docker run --env-file .env ...

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

CMD ["python", "app.py"]
