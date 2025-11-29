# Google Cloud Deployment Guide

## Prerequisites

1. **Google Cloud Account**: Make sure you have a Google Cloud account
2. **Google Cloud SDK**: Install the Google Cloud SDK on your local machine
3. **Project Setup**: Create a new project in Google Cloud Console

## Setup Steps

### 1. Install Google Cloud SDK

```bash
# Download and install from https://cloud.google.com/sdk/docs/install
# Or use package manager (macOS)
brew install google-cloud-sdk
```

### 2. Initialize Google Cloud SDK

```bash
gcloud init
# Follow the prompts to authenticate and select your project
```

### 3. Enable Required APIs

```bash
# Enable App Engine API
gcloud services enable appengine.googleapis.com

# Enable Cloud Build API (if needed)
gcloud services enable cloudbuild.googleapis.com
```

### 4. Create App Engine Application

```bash
# Create App Engine app in your preferred region
gcloud app create --region=us-central
```

### 5. Deploy Your Application

```bash
# Deploy to Google App Engine
gcloud app deploy

# Or deploy with specific configuration
gcloud app deploy app.yaml --version=v1 --no-promote
```

## Deployment Configuration

### Files Created:

1. **app.yaml** - Google App Engine configuration
   - Sets Python 3.9 runtime
   - Configures environment variables
   - Sets up auto-scaling

2. **main.py** - Entry point for App Engine
   - Imports your Flask app from scripts.serve_api
   - Configures the server to run on port 8080

3. **.gcloudignore** - Excludes unnecessary files from deployment
   - Ignores virtual environments, cache files, IDE files
   - Keeps deployment size minimal

## Environment Variables

The following environment variables are configured in `app.yaml`:
- `PORT`: 8080 (default port for App Engine)
- `RF_MODEL_PATH`: "models/random_forest_model.pkl"
- `ACC_MODEL_PATH`: "models/accelerometer_accident_detector.pkl"

## Testing After Deployment

Once deployed, your API will be accessible at:
```
https://YOUR-PROJECT-ID.appspot.com
```

Test your endpoints:
```bash
# Test the root endpoint
curl https://YOUR-PROJECT-ID.appspot.com/

# Test the predictions endpoint
curl https://YOUR-PROJECT-ID.appspot.com/predictions

# Test prediction endpoint
curl -X POST https://YOUR-PROJECT-ID.appspot.com/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "random_forest", "features": [1, 2, 3, 4, 5]}'
```

## Monitoring and Logs

```bash
# View application logs
gcloud app logs tail -s default

# View application status
gcloud app browse

# View traffic splitting
gcloud app services describe default
```

## Important Notes

1. **File Size**: The predictions.json file is large (81MB+). Make sure your App Engine instance has sufficient memory.

2. **Cold Starts**: The first request might be slower due to model loading.

3. **Scaling**: The auto-scaling configuration is set to handle 1-10 instances based on traffic.

4. **Costs**: Monitor your usage to avoid unexpected charges. App Engine has a free tier.

## Troubleshooting

If deployment fails:
1. Check the logs: `gcloud app logs read`
2. Verify all required files are present
3. Ensure models and data files are included in deployment
4. Check that requirements.txt includes all dependencies

## Alternative Deployment Options

If App Engine doesn't work well with large files, consider:
- **Cloud Run**: More flexible for containerized applications
- **Compute Engine**: Full VM control for large applications
- **Cloud Functions**: For smaller, event-driven functions