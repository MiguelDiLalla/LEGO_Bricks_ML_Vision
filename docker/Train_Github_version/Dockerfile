# Use official lightweight Python image as base
FROM python:3.10-slim AS builder

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./

# Install Python dependencies using cache mount for faster builds
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose necessary ports (if applicable)
EXPOSE 8080

# Define the entry point for the container
ENTRYPOINT ["python", "Train_cli.py"]
