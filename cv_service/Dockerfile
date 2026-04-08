# Use a Python image that supports OpenCV
FROM python:3.9-slim

# Install system dependencies for OpenCV and Kafka
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service code
COPY . .

# Run the CV service
CMD ["python", "main.py"]