FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-train.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Detect CUDA and install appropriate torch version
ARG TORCH_VERSION=2.4.1
RUN if python -c "import torch; print(torch.cuda.is_available())" | grep True; then \
        pip install torch==$TORCH_VERSION+cu118 --index-url https://download.pytorch.org/whl/cu118; \
    else \
        pip install torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/cpu; \
    fi


# Copy training scripts
COPY train_cli.py ./
COPY pipeline_train_docker.py ./

# Ensure proper execution permissions
RUN chmod +x train_cli.py

# Set entrypoint to the CLI
ENTRYPOINT ["python", "train_cli.py"]
