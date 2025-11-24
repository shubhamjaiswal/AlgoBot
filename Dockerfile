# Dockerfile
FROM python:3.11-slim

# set a working dir
WORKDIR /app

# copy requirements then install for layer caching
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    libxml2-dev \
    libxslt-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# create a non-root user for safety (optional)
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

# expose port if your bot serves an HTTP endpoint (optional)
# EXPOSE 8080

# default command - adapt to your bot entrypoint
CMD ["python", "runner.py"]
