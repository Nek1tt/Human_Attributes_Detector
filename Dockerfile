FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HAD_DATA_DIR=/app/var \
    HAD_DEVICE=cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --create-home app

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY SFSORT/SFSORT.py SFSORT/__init__.py SFSORT/LICENSE ./SFSORT/
RUN pip install --no-cache-dir ".[api,cpu]"

RUN mkdir -p /app/models /app/var/uploads /app/var/outputs \
    && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "silhouette_detector.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
