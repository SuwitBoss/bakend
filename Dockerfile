# Backend Dockerfile - Multi-stage build for optimal production deployment
# Stage 1: Build environment
FROM nvcr.io/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS builder

WORKDIR /build

# Configure apt for more reliable package downloads
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80retries && \
    echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/80retries && \
    echo 'Acquire::ftp::Timeout "120";' >> /etc/apt/apt.conf.d/80retries && \
    echo 'APT::Install-Recommends "false";' > /etc/apt/apt.conf.d/70recommends

# Install basic build tools and Python
RUN apt-get update && apt-get install -y \
    python3-dev python3-pip build-essential cmake \
    gcc g++ curl wget \
    && rm -rf /var/lib/apt/lists/*

# Install graphics and image processing libraries with retry logic
RUN for i in $(seq 1 3); do \
        apt-get update && \
        apt-get install -y \
        libgl1-mesa-glx libgl1-mesa-dev libglib2.0-0 \
        libsm6 libxext6 libxrender-dev libgomp1 && \
        rm -rf /var/lib/apt/lists/* && break || \
        echo "Retrying graphics libraries installation (attempt $i/3)..." && \
        sleep 5; \
    done

# Install video processing libraries with retry logic and split into smaller groups
# First group: basic codec libraries
RUN for i in $(seq 1 3); do \
        apt-get update && \
        apt-get install -y libavcodec-dev libavformat-dev libswscale-dev && \
        rm -rf /var/lib/apt/lists/* && break || \
        echo "Retrying codec libraries installation (attempt $i/3)..." && \
        sleep 5; \
    done

# Second group: GStreamer libraries
RUN for i in $(seq 1 3); do \
        apt-get update && \
        apt-get install -y libgtk-3-0 libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 && \
        rm -rf /var/lib/apt/lists/* && break || \
        echo "Retrying GStreamer libraries installation (attempt $i/3)..." && \
        sleep 5; \
    done

# Install image format libraries and OpenCV (split into smaller groups with retry logic)
# First install basic image libraries
RUN apt-get update && apt-get install -y \
    libjpeg-dev libpng-dev libtiff-dev libopenjp2-7-dev libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Then install OpenCV separately with retry logic
RUN for i in $(seq 1 3); do \
        apt-get update && \
        apt-get install -y python3-opencv && \
        rm -rf /var/lib/apt/lists/* && break || \
        echo "Retrying OpenCV installation (attempt $i/3)..." && \
        sleep 5; \
    done

# Install mathematical libraries and Boost with retry logic and in smaller groups
# First group: Atlas and GTK
RUN for i in $(seq 1 3); do \
        apt-get update && \
        apt-get install -y libatlas-base-dev libgtk-3-dev && \
        rm -rf /var/lib/apt/lists/* && break || \
        echo "Retrying Atlas/GTK installation (attempt $i/3)..." && \
        sleep 5; \
    done

# Second group: Boost and math libraries
RUN for i in $(seq 1 3); do \
        apt-get update && \
        apt-get install -y libboost-python-dev libopenblas-dev liblapack-dev pkg-config && \
        rm -rf /var/lib/apt/lists/* && break || \
        echo "Retrying Boost/math libraries installation (attempt $i/3)..." && \
        sleep 5; \
    done

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

# Install core dependencies with retry logic
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    for i in $(seq 1 3); do \
        pip install --no-cache-dir numpy pillow && break || \
        echo "Retrying numpy/pillow installation (attempt $i/3)..." && \
        sleep 5; \
    done

# Install OpenCV from pip instead of system package if python3-opencv fails
RUN pip install --no-cache-dir opencv-python || \
    pip install --no-cache-dir opencv-python-headless

# Install dlib - try multiple approaches for reliability
RUN pip install --no-cache-dir dlib==19.24.2 || \
    pip install --no-cache-dir dlib==19.24.1 || \
    pip install --no-cache-dir dlib

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Skip installing CUDA libraries if using nvcr.io image since they should be included
# Just verify they exist
RUN ldconfig -p | grep -E 'libcudnn|libcublas' || echo "Warning: Some CUDA libraries might be missing"

# Ensure GPU-enabled ONNX Runtime is installed (after requirements)
RUN pip uninstall -y onnxruntime && \
    pip install --no-cache-dir onnxruntime-gpu==1.16.0 || \
    pip install --no-cache-dir onnxruntime==1.16.0

# Create symbolic links for CUDA libraries if needed (only if specific version required)
RUN if [ -d /usr/local/cuda/lib64 ]; then \
    echo "CUDA libraries found at: /usr/local/cuda/lib64" && \
    ls -la /usr/local/cuda/lib64/libcublas* && \
    if [ ! -f /usr/local/cuda/lib64/libcublasLt.so.11 ] && [ -f /usr/local/cuda/lib64/libcublasLt.so ]; then \
        ln -s /usr/local/cuda/lib64/libcublasLt.so /usr/local/cuda/lib64/libcublasLt.so.11; \
    fi; \
else \
    echo "CUDA not available in standard location, checking elsewhere"; \
    find / -name "libcudnn.so*" -o -name "libcublas.so*" 2>/dev/null || echo "No CUDA libraries found"; \
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
