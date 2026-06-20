FROM python:3.12-slim

WORKDIR /app

# Install build tools needed by some Python packages (e.g. aiosqlite C extensions).
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (separate layer for cache efficiency).
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy application source.
COPY app/ ./app/

# Create default runtime directories.
RUN mkdir -p /app/sandbox /app/data

# Default command: run the UI server.
# Override in docker-compose.yml per service (e.g. consumer uses python -m app.services.mq.runner).
CMD ["uvicorn", "app.ui.api:app", "--host", "0.0.0.0", "--port", "8080"]
