FROM python:3.12-slim

WORKDIR /app

# Install system deps for lxml/beautifulsoup
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2-dev libxslt1-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

# Install uv and dependencies
RUN pip install uv && uv sync --frozen --no-dev

COPY . .

EXPOSE 6847

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "6847"]
