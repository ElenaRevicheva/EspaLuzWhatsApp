# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (audio + OCR support)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        libsm6 \
        libxext6 \
        libxrender-dev \
        curl \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Set environment variables (for runtime compatibility)
ENV PYTHONUNBUFFERED=1

# Expose the Flask default port for Railway or Meta Webhook registration
EXPOSE 8080

# Start the Flask WhatsApp bridge
CMD ["python", "espaluz_bridge.py"]
