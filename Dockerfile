# =======================
# 🏗 Base Image & Setup
# =======================
FROM python:3.10-slim AS base

# Set working directory inside the container
WORKDIR /app

# Ensure system packages are up to date
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# =======================
# 📦 Install Dependencies
# =======================
FROM base AS builder

# Copy dependency files
COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# =======================
# 🚀 Final Image
# =======================
FROM base

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Ensure scripts use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy the source code
COPY . .

# =======================
# 🔒 Secure Kaggle Credentials (Secret Mounting)
# =======================
# This will be mounted at runtime
RUN mkdir -p /root/.kaggle && chmod 700 /root/.kaggle

# =======================
# 🗂️ Data Storage (Docker Volumes)
# =======================
VOLUME /app/data

# =======================
# 🏁 Entrypoint & Default Command
# =======================
ENTRYPOINT ["python", "-m", "scripts.cli"]
CMD ["--help"]
