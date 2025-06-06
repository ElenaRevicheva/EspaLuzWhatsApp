# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all your files
COPY . .

# Install pip packages
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose Railway port
ENV PORT=5000
EXPOSE $PORT

# Start the Flask server
CMD ["python", "espaluz_bridge.py"]
