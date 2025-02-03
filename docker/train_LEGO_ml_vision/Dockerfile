FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (with cleanup)
RUN apt update && apt install -y --no-install-recommends \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements file
COPY requirements-train.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Detect CUDA and install appropriate torch version (optimized)
ARG TORCH_VERSION=2.4.1
RUN if python -c "import torch; print(torch.cuda.is_available())" | grep True; then \
        pip install --no-cache-dir torch==$TORCH_VERSION+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
        pip uninstall -y triton; \
    else \
        pip install --no-cache-dir torch==$TORCH_VERSION+cpu; \
    fi

# Remove unnecessary cached files
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy training scripts
COPY train_cli.py ./train_cli.py
COPY pipeline_train_docker.py ./pipeline_train_docker.py

# Ensure execution permissions
RUN chmod +x train_cli.py

# Set entrypoint to the CLI
ENTRYPOINT ["python", "train_cli.py"]
