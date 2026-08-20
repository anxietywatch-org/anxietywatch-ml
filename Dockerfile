# syntax=docker/dockerfile:1

# AnxietyWatch ML inference service — Azure Container Apps (Linux) ready.
#
# The trained model artifact is NOT baked into the image: it is supplied at
# runtime via ANXIETYWATCH_MODEL_PATH (e.g. a volume mount or ACA secret).
# Production startup is fail-fast: without a loadable artifact the process
# exits instead of serving traffic with no model.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANXIETYWATCH_REQUIRE_MODEL=true \
    PORT=8000

WORKDIR /app

# Install the project and its runtime dependencies reproducibly.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Secondary guard: container-level health (ACA also probes /health itself).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# Bind 0.0.0.0 so Azure Container Apps ingress reaches the service, honoring
# the PORT convention used by the Container Apps platform.
CMD ["sh", "-c", "exec uvicorn anxietywatch_ml.serving.app:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1"]