# Use NVIDIA CUDA 12.4 base image (compatible with driver 596.49 / CUDA 13.2)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3/pip point to 3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Optional vLLM install (GPU-only) — default true now that we have CUDA base
ARG INSTALL_VLLM=true

# Copy requirements
COPY drag_llm_service/requirements.txt .
COPY drag_llm_service/requirements-vllm.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_VLLM" = "true" ]; then pip install --no-cache-dir -r requirements-vllm.txt; fi

# Install CUDA compiler + ninja (needed for FlashInfer JIT kernel compilation)
# This is a separate layer so it doesn't invalidate the cached pip install above
RUN apt-get update && apt-get install -y \
    cuda-nvcc-12-4 \
    libcurand-dev-12-4 \
    cuda-libraries-dev-12-4 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy drag_python_client library
COPY drag_python_client /app/drag_python_client

# Copy drag_contract for ABI access
COPY drag_contract /app/drag_contract

# Copy application code
COPY drag_llm_service /app

# Expose port
EXPOSE 9000

# Run the server
CMD ["python", "app/server.py"]
