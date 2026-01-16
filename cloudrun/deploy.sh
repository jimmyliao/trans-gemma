#!/bin/bash
set -e

# TranslateGemma Cloud Run Deployment Script
# This script deploys the TranslateGemma API to Google Cloud Run with GPU support

# Configuration
export PROJECT_ID="${PROJECT_ID:-gde-jimmyliao-redeem}"
export REGION="${REGION:-us-central1}"
export SERVICE_NAME="${SERVICE_NAME:-translategemma-4b}"
export IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "======================================"
echo "TranslateGemma Cloud Run Deployment"
echo "======================================"
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service Name: ${SERVICE_NAME}"
echo "Image: ${IMAGE_NAME}"
echo "======================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project
echo "Setting GCP project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com

# Build container image
echo "Building container image..."
docker build -t ${IMAGE_NAME} .

# Push to Google Container Registry
echo "Pushing image to GCR..."
docker push ${IMAGE_NAME}

# Deploy to Cloud Run with GPU
echo "Deploying to Cloud Run with GPU..."
gcloud beta run deploy ${SERVICE_NAME} \
    --image=${IMAGE_NAME} \
    --platform=managed \
    --region=${REGION} \
    --port=8080 \
    --cpu=4 \
    --memory=16Gi \
    --timeout=300 \
    --no-cpu-throttling \
    --gpu=1 \
    --gpu-type=nvidia-l4 \
    --max-instances=3 \
    --min-instances=0 \
    --allow-unauthenticated

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region=${REGION} \
    --format='value(status.url)')

echo "======================================"
echo "Deployment completed successfully!"
echo "======================================"
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test the API:"
echo "curl -X POST ${SERVICE_URL}/translate \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"text\": \"Hello, world!\", \"target_lang\": \"Traditional Chinese\"}'"
echo "======================================"
