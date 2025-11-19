#!/bin/bash

# Exit on error
set -e

APP_NAME="captchav2"
REGION="us-central1"  # You can change this to your preferred region

echo "Deploying $APP_NAME to Google Cloud Run..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Submit the build to Cloud Build
echo "Building container image..."
gcloud builds submit --tag gcr.io/$(gcloud config get-value project)/$APP_NAME

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $APP_NAME \
    --image gcr.io/$(gcloud config get-value project)/$APP_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated

echo "Deployment complete!"
