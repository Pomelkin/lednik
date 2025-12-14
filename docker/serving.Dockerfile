FROM ghcr.io/astral-sh/uv:python3.13-trixie

WORKDIR /api

COPY ./ ./

ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
uv sync --locked --no-default-groups --group serving --compile-bytecode \
useradd --no-create-home --shell /bin/bash appuser \
chown -R appuser:appuser /api

USER appuser

ENV PYTHONPATH=/api:$PYTHONPATH \
    MODEL_ID="" \
    PORT=8080

ENTRYPOINT [ "/bin/sh", "-c", "exec uv run python ./lednik/serving/api.py --port ${PORT:-8080} --model-id ${MODEL_ID}" ]
