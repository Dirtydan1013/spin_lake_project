# ================================================================
# Stage 1: Builder — compile the C++ extension with OpenMP
# ================================================================
FROM python:3.12-slim AS builder

# Install system build tools (compiler, cmake, ninja, OpenMP dev headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ cmake ninja-build libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python build-time dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir pybind11 \
 && pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

# Copy C++ source and CMakeLists
COPY CMakeLists.txt .
COPY csrc/ ./csrc/

# Build the C++ extension (produces qaqmc_cpp*.so)
RUN cmake -S . -B build_dir -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_STRIP="" \
        -DPYTHON_EXECUTABLE=$(which python3) \
    && cmake --build build_dir --config Release -j \
    && cp build_dir/qaqmc_cpp*.so .


# ================================================================
# Stage 2: Runtime — lean image containing only what's needed
# ================================================================
FROM python:3.12-slim

# Only the OpenMP runtime library (not the heavy dev headers or compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages from pre-built wheels (fast, no compilation)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy the compiled C++ extension module from builder stage
COPY --from=builder /build/qaqmc_cpp*.so /app/

# Copy Python source code
COPY src/ ./src/
COPY test.py .

# Ensure output data directory exists
RUN mkdir -p /app/data

# Default: test.py — override at `docker run` time with any script you like
CMD ["python", "test.py"]
