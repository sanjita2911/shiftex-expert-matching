FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ca-certificates \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    libmagickwand-dev \
    imagemagick \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Repo code
COPY . /app

# Ensure local imports work when running scripts directly
ENV PYTHONPATH=/app

# Default is a no-op; k8s yaml sets the command for server/client pods.
CMD ["python3", "-c", "print('ShiftEx image built. Set command to run server/server.py or run_client.py')"]