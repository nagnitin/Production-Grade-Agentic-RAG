# ==========================================
# Builder Stage
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging configuration
COPY pyproject.toml README.md ./

# Install project dependencies into a local directory
RUN pip install --no-cache-dir --user .

# ==========================================
# Production Stage
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime dependencies (like libpq for PostgreSQL connection)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy codebase
COPY src/ ./src/

# Expose port for API
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production

# Run FastAPI app via Uvicorn factory
CMD ["uvicorn", "src.api.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
