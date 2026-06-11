# Use an official lightweight Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MULTISHIELD_CHECKPOINT_DIR="experiments/checkpoints" \
    PORT=8000 \
    HOST="0.0.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies needed for some libraries (e.g., git or build-essentials if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# Note: To keep the container size small and run efficiently on CPU-only free tier hosting platforms,
# we install the CPU-only version of PyTorch and Torchvision.
RUN pip install --no-cache-dir pip --upgrade && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy the rest of the application files
COPY src/ ./src/
COPY experiments/checkpoints/ ./experiments/checkpoints/
COPY static/ ./static/
COPY app.py .

# Expose port
EXPOSE 8000

# Run the healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
