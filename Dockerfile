# === Base Python image ===
FROM python:3.11-slim

# === Environment Variables ===
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata

# === Working Directory ===
WORKDIR /app

# === System Dependencies ===
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# === Copy dependencies first for caching ===
COPY requirements.txt .

# === Install Python dependencies ===
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# === Copy project files ===
COPY . .

# === Healthcheck (optional, ECS compatible) ===
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Add supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# === Expose app port ===
EXPOSE 8000

# === Default command ===
CMD ["python", "runner.py"]
