# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
#Trigger deployment
# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads temp sessions logs embeddings face_database

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV PORT=10000

# Expose port (Railway will override with $PORT)
EXPOSE 10000

# Health check - increased timeout for model loading
HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:$PORT/health', timeout=10)" || exit 1

# Run gunicorn - Railway optimized (2 workers to prevent OOM)
# Railway auto-sets $PORT, WEB_CONCURRENCY from .env
CMD gunicorn -w ${WEB_CONCURRENCY:-2} -b 0.0.0.0:${PORT:-10000} \
    --timeout ${GUNICORN_TIMEOUT:-120} \
    --graceful-timeout 120 \
    --keep-alive 5 \
    --max-requests ${GUNICORN_MAX_REQUESTS:-500} \
    --max-requests-jitter ${GUNICORN_MAX_REQUESTS_JITTER:-50} \
    --worker-class sync \
    --worker-tmp-dir /dev/shm \
    --preload \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    main:app
