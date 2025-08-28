# Multi-stage Docker build for IPE (Integrated Phenotypic Evolution) platform
# Production container size optimized to be < 1GB

# Stage 1: Python build environment
FROM python:3.12-slim as python-builder

WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Node.js build environment
FROM node:18-alpine as web-builder

WORKDIR /build/web

# Install dependencies (if package.json exists)
COPY web/package*.json ./
RUN if [ -f package.json ]; then npm ci --only=production; fi

# Copy web source and build (if exists)
COPY web/ .
RUN if [ -f package.json ]; then npm run build 2>/dev/null || echo "No build script found"; fi

# Stage 3: Production runtime
FROM python:3.12-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    liblapack3 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r ipe && useradd -r -g ipe ipe

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=python-builder /root/.local /home/ipe/.local

# Copy web build from builder stage (if exists)
COPY --from=web-builder /build/web/dist /app/web/dist 2>/dev/null || echo "No web build to copy"

# Copy application code
COPY ipe/ ./ipe/
COPY scripts/ ./scripts/
COPY documentation/ ./documentation/
COPY demo_core_001.py .
COPY requirements.txt .

# Set Python path and user paths
ENV PATH="/home/ipe/.local/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Change ownership to non-root user
RUN chown -R ipe:ipe /app

# Switch to non-root user
USER ipe

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import ipe; print('IPE module loaded successfully')" || exit 1

# Default command
CMD ["python", "-m", "ipe.api.server"]

# Stage 4: Development environment
FROM production as development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python development packages
COPY requirements.txt .
RUN pip install --no-cache-dir pytest ipython jupyter

# Install pre-commit hooks if available
RUN if [ -f .pre-commit-config.yaml ]; then pip install pre-commit; fi

USER ipe

# Development command with auto-reload
CMD ["python", "-m", "ipe.api.server", "--debug", "--reload"]