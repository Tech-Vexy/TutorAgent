FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for psycopg and other libs
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for ChromaDB persistence
RUN mkdir -p chroma_db

# Expose the port
EXPOSE 8000

# Run the application
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
