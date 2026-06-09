FROM nvidia/cuda:12.8.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl build-essential ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build-time user mapping (pass via --build-arg)
ARG HOST_UID=""
ARG HOST_GID=""
ARG HOST_USER=""

RUN test -n "$HOST_UID" || (echo "HOST_UID is required" && exit 1); \
    test -n "$HOST_GID" || (echo "HOST_GID is required" && exit 1); \
    test -n "$HOST_USER" || (echo "HOST_USER is required" && exit 1) && \
    groupadd -g ${HOST_GID} ${HOST_USER} 2>/dev/null || true && \
    useradd -l -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_USER} 2>/dev/null || true

USER ${HOST_USER}
WORKDIR /home/${HOST_USER}/app

RUN curl -LsSf https://astral.sh/uv/install.sh -o install_uv.sh && \
    chmod +x install_uv.sh && \
    bash install_uv.sh && rm install_uv.sh
ENV PATH="/home/${HOST_USER}/.local/bin:${PATH}"

CMD [ "/bin/bash" ]
