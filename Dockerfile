# Dockerfile
FROM python:3.10-slim

# Prevent Python from writing .pyc files
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data requirements
RUN python -c "import nltk; nltk.download('punkt')"

# Copy application source code
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run FastAPI backend via Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]