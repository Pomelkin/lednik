FROM nvidia/cuda:12.8.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git=1:2.34.1-1ubuntu1.17 \
    curl=7.81.0-1ubuntu1.25 \
    build-essential=12.9ubuntu3 \
    ninja-build=1.10.1-1 \
    ca-certificates=20260601~22.04.1 \
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
WORKDIR /home/${HOST_USER}/lednik

RUN curl -LsSf https://astral.sh/uv/install.sh -o install_uv.sh && \
    chmod +x install_uv.sh && \
    bash install_uv.sh && rm install_uv.sh && \
    mkdir -p /home/${HOST_USER}/.cache/uv

ENV PATH="/home/${HOST_USER}/.local/bin:${PATH}" \
    PYTHONPATH="/home/${HOST_USER}/lednik" \
    UV_CACHE_DIR="/home/${HOST_USER}/.cache/uv" \
    UV_LINK_MODE=copy

# Dependencies first: this layer is cached until pyproject/lock/wheels change
COPY --chown=${HOST_UID}:${HOST_GID} pyproject.toml uv.lock ./
COPY --chown=${HOST_UID}:${HOST_GID} kostyl_toolkit/ kostyl_toolkit/
COPY --chown=${HOST_UID}:${HOST_GID} wheels/ wheels/
# distill group is needed for ClearML model-id resolution in LednikServer
RUN uv sync --frozen --no-install-project --no-default-groups \
    --group serving --group distill --group fast-attn

# Project code last: changing it only invalidates this cheap layer
COPY --chown=${HOST_UID}:${HOST_GID} lednik/ lednik/
RUN uv sync --frozen --no-default-groups \
    --group serving --group distill --group fast-attn

ENV FLA_CONV_BACKEND=cuda

EXPOSE 8000

# CLI arguments (--model, --tokenizer, ...) are appended at runtime
# via `docker run <image> <args>` or `command:` in compose
ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "lednik.serving.server"]
