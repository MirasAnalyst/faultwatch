# FaultWatch scoring API.
#   docker build -t faultwatch .
#   docker run -p 8000:8000 -v $PWD/models:/app/models faultwatch
# Models are trained outside the image (python run.py configs/*.yaml) and
# mounted, so one image serves any retrained version.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt
COPY faultwatch ./faultwatch
COPY api.py dashboard.py run.py ./
COPY configs ./configs
COPY corpus ./corpus
RUN useradd -m fw && mkdir -p /app/models /app/monitoring && chown -R fw /app
USER fw
ENV FAULTWATCH_ROOT=/app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
