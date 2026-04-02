FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configs
COPY src/ ./src/
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/
COPY static/ ./static/

# Create logs directory
RUN mkdir -p logs

# Create data directory for S3 downloads
RUN mkdir -p data/raw data/processed

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with environment variables (pass via docker run -e)
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
