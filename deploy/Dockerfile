# --- Stage 1: Builder ---
FROM ubuntu:24.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
# GCC 13 is default in Ubuntu 24.04
RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  python3 \
  python3-pip \
  python3-venv \
  python3-dev \
  git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source
COPY . .

# Build wheel
# We reference the root to pick up pyproject.toml and CMakeLists.txt
RUN python3 -m pip install --break-system-packages build scikit-build-core
RUN python3 -m build --wheel --outdir dist/ .

# --- Stage 2: Runtime ---
FROM python:3.12-slim AS runtime

# Create non-root user
RUN groupadd -r aeon && useradd -r -g aeon -m -d /home/aeon aeon

WORKDIR /home/aeon/app

# Install runtime dependencies if any (none for now)
# Copy wheel from builder
COPY --from=builder /build/dist/*.whl /tmp/

# Install the wheel
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Switch to non-root user
USER aeon

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import aeon_py.core; print('Health: OK')" || exit 1

# Default command
CMD ["python", "-c", "import aeon_py.core; print(aeon_py.core.get_build_info())"]
