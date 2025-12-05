# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

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

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:10000/health', timeout=5)" || exit 1

# Run gunicorn
CMD gunicorn -w 2 -b 0.0.0.0:$PORT \
    --timeout 300 \
    --graceful-timeout 300 \
    --keep-alive 5 \
    --max-requests 100 \
    --worker-class sync \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    main:app
