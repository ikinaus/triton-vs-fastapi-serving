FROM python:3.12

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY .streamlit ./.streamlit
COPY theme.py .
COPY streamlit_app.py .

CMD [ ".venv/bin/streamlit", "run", "streamlit_app.py" ]
