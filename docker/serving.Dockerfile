FROM ghcr.io/astral-sh/uv:python3.13-trixie

WORKDIR /opt/app

COPY ./ ./

ENV UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120

RUN --mount=type=cache,target=/root/.cache/uv \
uv sync --locked --no-default-groups --group serving --compile-bytecode && \
useradd --create-home --shell /bin/bash appuser && \
chown -R appuser:appuser /opt/app

USER appuser

ENV PYTHONPATH=/opt/app:$PYTHONPATH \
    MODEL_ID="" \
    TOKENIZER_ID="" \
    CLEARML_API_ACCESS_KEY="" \
    CLEARML_API_SECRET_KEY="" \
    CLEARML_API_HOST="" \
    AWS_ACCESS_KEY_ID="" \
    AWS_SECRET_ACCESS_KEY="" \
    PORT=8080

ENTRYPOINT [ "/bin/sh", "-c", "exec uv run --no-sync python ./lednik/serving/api.py --port ${PORT:-8080} --model-id ${MODEL_ID} --tokenizer-id ${TOKENIZER_ID:-''}" ]
