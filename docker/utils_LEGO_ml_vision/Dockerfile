FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-utils.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set up Kaggle authentication
RUN mkdir -p /root/.kaggle && \
    if [ -f "/run/secrets/kaggle_secret" ]; then \
        cp /run/secrets/kaggle_secret /root/.kaggle/kaggle.json && \
        chmod 600 /root/.kaggle/kaggle.json; \
    fi

# Copy utility scripts
COPY pipeline-utils_docker.py ./
COPY utils_cli.py ./

# Ensure proper execution permissions
RUN chmod +x utils_cli.py

# Set entrypoint to the CLI
ENTRYPOINT ["python", "utils_cli.py"]
