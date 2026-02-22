FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    unzip \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && python -m pip install --upgrade pip "setuptools<81" wheel

WORKDIR /workspace
COPY . /workspace

# Core tracker package
RUN pip install -r requirements.txt && pip install -e . --no-build-isolation

# Runtime dependencies for nuScenes conversion + tracking.
RUN pip install \
    nuscenes-devkit==1.1.10 \
    pyquaternion==0.9.9 \
    motmetrics==1.1.3 \
    pandas==1.5.3

# Optional Sparse4D detection runtime dependencies (Linux x86 + GPU oriented).
ARG INSTALL_SPARSE4D_DEPS=0
RUN if [ "$INSTALL_SPARSE4D_DEPS" = "1" ]; then \
      pip install \
        torch==1.13.0+cu116 \
        torchvision==0.14.0+cu116 \
        torchaudio==0.13.0 \
        --extra-index-url https://download.pytorch.org/whl/cu116 && \
      pip install \
        numpy==1.23.5 \
        mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html \
        mmdet==2.28.2 \
        yapf==0.33.0 \
        tensorboard==2.14.0 \
        urllib3==1.26.16 ; \
    fi

# Optional: compile Sparse4D op if repo exists after fetch script.
RUN chmod +x /workspace/nuscenes_runtime/scripts/*.sh /workspace/docker/*.sh || true

CMD ["bash"]
