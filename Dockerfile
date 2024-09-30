# Base image
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # python3.12 \
    python3-pip \
    git \
    wget \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    software-properties-common

# Download and install Ollama
# RUN curl -fsSL https://ollama.com/install.sh | sh

# Add Ollama to the PATH if it's not already in a system path
# ENV PATH="$PATH:/usr/local/bin"

# Clean up APT when done
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# # Expose any ports if needed (based on how Ollama serves your app)
# EXPOSE 11400

# Create and set the working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt -v

# Expose port for Streamlit
# EXPOSE 8501

# Copy the application files
COPY . .
COPY nvidia_entrypoint.sh /opt/nvidia/

# # # Expose the default port used by Ollama
# EXPOSE 11400

# # Command to start Ollama
# CMD ["ollama", "serve", "--host", "0.0.0.0"]
# CMD ["ollama", "pull", "nomic-embed-text:latest"]
# CMD ["ollama", "pull", "llama3.1:8b"]

# Run the Streamlit application
# CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]

