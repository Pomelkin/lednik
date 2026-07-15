FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

ARG HOST_UID=""
ARG HOST_GID=""
ARG HOST_USER=""
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""

# System deps
RUN test -n "$HOST_UID" || (echo "HOST_UID is required" && exit 1); \
    test -n "$HOST_GID" || (echo "HOST_GID is required" && exit 1); \
    test -n "$HOST_USER" || (echo "HOST_USER is required" && exit 1); \
    apt-get update -qq -o=Dpkg::Use-Pty=0 && \
    apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
        git=1:2.34.1-1ubuntu1.17 \
        git-lfs=3.0.2-1ubuntu0.3 \
        openssh-client=1:8.9p1-3ubuntu0.15 \
        curl=7.81.0-1ubuntu1.25 \
        ca-certificates=20240203~22.04.1 \
        libgomp1=12.3.0-1ubuntu1~22.04.3 \
        libglib2.0-0=2.72.4-0ubuntu2.9 \
        libgl1=1.4.0-1 \
        ffmpeg=7:4.4.2-0ubuntu0.22.04.1 \
        libsm6=2:1.2.3-1build2 \
        libxext6=2:1.3.4-1build1 \
        pkg-config=0.29.2-1ubuntu3 \
        cmake=3.22.1-1ubuntu1.22.04.2 \
        ninja-build=1.10.1-1 \
        fish=3.3.1+ds-3 && \
    rm -rf /var/lib/apt/lists/* && \
   (groupadd -g ${HOST_GID} ${HOST_USER} || echo "Group ${HOST_GID} already exists") && \
    (useradd -l -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_USER} || echo "User ${HOST_USER} already exists")

USER ${HOST_USER}
WORKDIR /home/${HOST_USER}/lednik

RUN curl -LsSf https://astral.sh/uv/install.sh -o install_uv.sh && \
    chmod +x install_uv.sh && \
    bash install_uv.sh && rm install_uv.sh && \
    mkdir -p /home/${HOST_USER}/.local/share \
             /home/${HOST_USER}/lednik/.venv \
             /home/${HOST_USER}/.cache

ENV PATH="/home/${HOST_USER}/.local/bin:${PATH}" \
    PYTHONPATH="/home/${HOST_USER}/lednik" \
    UV_CACHE_DIR="/home/${HOST_USER}/.cache/uv" \
    UV_LINK_MODE=copy

CMD [ "/bin/fish" ]
