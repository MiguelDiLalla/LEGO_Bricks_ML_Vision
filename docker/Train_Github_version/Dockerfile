# Use official lightweight Python image as base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose necessary ports (if applicable)
EXPOSE 8080

# Define the entry point for the container
ENTRYPOINT ["python", "Train_cli.py"]
