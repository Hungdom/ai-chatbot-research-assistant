FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set Python path to include the src directory
ENV PYTHONPATH=/app

# Create necessary directories
RUN mkdir -p /app/logs /app/src/crawler/data

# Set permissions
RUN chmod -R 755 /app

CMD ["python", "src/run_all_topics.py"] 