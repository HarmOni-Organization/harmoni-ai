# Use official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/my_modules /app/tests

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest pytest-cov gunicorn

# Copy application files
COPY app.py .
COPY my_modules/ ./my_modules/
COPY tests/ ./tests/

# Set proper permissions
RUN chmod -R 755 /app

# Expose the port Flask is running on
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "app:app"]