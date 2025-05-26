FROM python:3.11-slim

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temp directory
RUN mkdir -p /tmp/videobot

# Expose port for Railway
EXPOSE 8000

# Health check endpoint
RUN echo 'from http.server import HTTPServer, BaseHTTPRequestHandler\nimport threading\nclass HealthHandler(BaseHTTPRequestHandler):\n    def do_GET(self):\n        if self.path == "/health":\n            self.send_response(200)\n            self.end_headers()\n            self.wfile.write(b"OK")\nserver = HTTPServer(("", 8000), HealthHandler)\nthreading.Thread(target=server.serve_forever, daemon=True).start()' > health.py

# Run the bot
CMD ["python", "bot.py"]
