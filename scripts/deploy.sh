#!/usr/bin/env bash
# ==========================================
# Deployment Orchestrator Script
# ==========================================
set -euo pipefail

# Text color formatting constants
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0;3b' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Check dependencies
log_info "Verifying developer environment dependencies..."
for cmd in gcloud terraform docker; do
    if ! command -v "$cmd" &> /dev/null; then
        log_error "Dependency '$cmd' is missing. Please install it first."
        exit 1
    fi
done
log_success "All build tools verified."

# 2. Login to GCP
log_info "Authenticating with Google Cloud..."
gcloud auth application-default login --quiet || log_info "Default credentials already configured."

# 3. Apply Terraform Infrastructure
log_info "Initializing and applying Terraform infrastructure..."
cd infrastructure/terraform
terraform init
terraform apply -var-file="environments/dev.tfvars" -auto-approve

# Extract output values
REGION=$(terraform output -raw region 2>/dev/null || echo "us-central1")
PROJECT_ID=$(gcloud config get-value project)
REPO_NAME=$(terraform output -raw artifact_registry_repository | awk -F '/' '{print $NF}' 2>/dev/null || echo "agentic-rag-dev-repo")
APP_NAME="agentic-rag"

log_success "Infrastructure provisioned."
cd ../..

# 4. Configure Secret Manager values
log_info "Setting up system secrets in GCP Secret Manager..."
read -sp "Enter Portkey API Key (Press Enter to skip/leave unchanged): " PORTKEY_KEY
echo ""
if [ -n "$PORTKEY_KEY" ]; then
    echo -n "$PORTKEY_KEY" | gcloud secrets versions add "${APP_NAME}-dev-portkey-key" --data-file=-
    log_success "Portkey API Key secret version added."
fi

# 5. Build and deploy services using Cloud Build
log_info "Submitting project workspace to Google Cloud Build..."
gcloud builds submit --config=cloudbuild.yaml \
    --substitutions="_REGION=${REGION},_REPO_NAME=${REPO_NAME},_APP_NAME=${APP_NAME}" \
    .

log_success "CI/CD Deployment process finished successfully!"
log_info "Endpoints can be queried from the Cloud Run Dashboard."
