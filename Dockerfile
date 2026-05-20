# syntax=docker/dockerfile:1.6
#
# MES — Motor Engagement Signal — single-container HF Space image.
# - Stage 1: build the React frontend
# - Stage 2: install Python deps (CPU-only) and copy the built frontend
#
# Exposes port 7860 (HF Space convention).

# ---------- Stage 1: frontend build ----------
FROM node:20-bookworm-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund
COPY frontend/ ./
COPY backend/ /app/backend/
RUN npm run build  # writes into /app/backend/app/static


# ---------- Stage 2: runtime ----------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    MES_DATA=/data \
    PATH=/home/mes/.local/bin:$PATH

# System libs for MNE + WeasyPrint + Pango/Cairo.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libffi8 \
      libgdk-pixbuf2.0-0 fonts-dejavu \
      libopenblas0 liblapack3 \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 mes
WORKDIR /home/mes/app

# Copy backend + library code.
COPY pyproject.toml README.md LICENSE ./
COPY mes_core/ ./mes_core/
COPY backend/ ./backend/
COPY notebooks/ ./notebooks/

# Install CPU-only torch via the index then the rest from pyproject.
# We deliberately skip the `train` extra (huge torch GPU builds) and pin CPU torch.
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.2.2+cpu" "torchvision==0.17.2+cpu" || true
RUN pip install -e ".[backend]"

# Bring the built frontend in.
COPY --from=frontend /app/backend/app/static ./backend/app/static

# Persistent volume.
RUN mkdir -p /data && chown -R mes:mes /home/mes /data
USER mes

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:${PORT}/api/healthz || exit 1

CMD ["python", "-m", "backend.app.main"]
