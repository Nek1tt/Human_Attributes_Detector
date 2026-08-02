FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

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
COPY requirements ./requirements
COPY src ./src
COPY SFSORT/SFSORT.py SFSORT/__init__.py SFSORT/LICENSE ./SFSORT/
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
       --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements/cpu.txt

RUN mkdir -p /app/models /app/var/uploads /app/var/outputs \
    && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "silhouette_detector.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
