# Backend Dockerfile - Multi-stage build for optimal production deployment
# Stage 1: Build environment
FROM nvidia/cuda:12.2-devel-ubuntu22.04 AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y \
    python3-dev python3-pip build-essential cmake \
    gcc g++ curl wget \
    libgl1-mesa-glx libgl1-mesa-dev libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libgomp1 \
    libgtk-3-0 libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
    python3-opencv libjpeg-dev libpng-dev libtiff-dev \
    libopenjp2-7-dev libboost-all-dev libx11-dev \
    libatlas-base-dev libgtk-3-dev libboost-python-dev \
    libopenblas-dev liblapack-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for ONNX Runtime and PyTorch - Optimize for multi-core
ENV OMP_NUM_THREADS=12
ENV OPENBLAS_NUM_THREADS=12
ENV MKL_NUM_THREADS=12
ENV VECLIB_MAXIMUM_THREADS=12
ENV NUMEXPR_NUM_THREADS=12
ENV ONNXRUNTIME_LOG_LEVEL=3
ENV TORCH_HOME=/app/.torch
ENV TORCH_VISION_CACHE=/app/.torchvision

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in stages for better caching and error handling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip install --no-cache-dir numpy opencv-python pillow

# Install dlib - try multiple approaches for reliability
RUN pip install --no-cache-dir dlib==19.24.2 || \
    pip install --no-cache-dir dlib==19.24.1 || \
    pip install --no-cache-dir dlib

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install CUDA libraries needed for ONNX Runtime GPU (if available)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcublas11 libcublaslt11 libcublas-dev libcudnn8 || echo "CUDA libraries not available, continuing without them"

# Ensure GPU-enabled ONNX Runtime is installed (after requirements)
RUN pip uninstall -y onnxruntime && \
    pip install --no-cache-dir onnxruntime-gpu==1.16.0 || \
    pip install --no-cache-dir onnxruntime==1.16.0

# Create symbolic links for CUDA libraries if needed (only if CUDA is available)
RUN if [ -d /usr/local/cuda/lib64 ]; then \
    if [ ! -f /usr/local/cuda/lib64/libcublasLt.so.11 ] && [ -f /usr/local/cuda/lib64/libcublasLt.so ]; then \
        ln -s /usr/local/cuda/lib64/libcublasLt.so /usr/local/cuda/lib64/libcublasLt.so.11; \
    fi; \
else \
    echo "CUDA not available, skipping symbolic links"; \
fi

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with gunicorn for production performance
# Use UvicornWorker for FastAPI (ASGI) compatibility
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-c", "gunicorn.conf.py"]
