# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Download model at build time (optional - can also be done at runtime)
# Uncomment the following lines to pre-download the model:
# RUN python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
#     model_id='google/translategemma-4b-it'; \
#     AutoTokenizer.from_pretrained(model_id); \
#     AutoModelForCausalLM.from_pretrained(model_id)"

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
