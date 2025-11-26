# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv

# Move to ubuntu user home
WORKDIR /home/ubuntu/

# Clone llm_wrappers
RUN git clone https://github.com/aitor-ibarguren/llm_wrappers.git

# Install python3-venv (for virtual environments)
RUN apt-get update && apt-get install -y python3-venv

# Create a virtual environment
RUN python3 -m venv /home/ubuntu/venv

# Install dependencies & llm_wrappers inside the virtual environment
RUN cd llm_wrappers && /home/ubuntu/venv/bin/pip3 install --no-cache-dir -r requirements.txt && /home/ubuntu/venv/bin/pip3 install --no-cache-dir -e .