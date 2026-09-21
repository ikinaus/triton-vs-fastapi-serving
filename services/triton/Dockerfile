FROM nvcr.io/nvidia/tritonserver:26.06-py3

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /env

COPY pyproject.toml uv.lock ./

RUN uv export --frozen --no-dev --no-hashes -o requirements.txt \
 && uv pip install --system --break-system-packages -r requirements.txt \
 && rm requirements.txt
