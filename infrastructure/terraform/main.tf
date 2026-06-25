terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ==========================================
# VPC Network Submodule
# ==========================================
module "vpc" {
  source      = "./modules/vpc"
  app_name    = var.app_name
  environment = var.environment
  region      = var.region
}

# ==========================================
# Cloud SQL Submodule (PostgreSQL)
# ==========================================
module "cloud_sql" {
  source         = "./modules/cloud_sql"
  app_name       = var.app_name
  environment    = var.environment
  region         = var.region
  network_id     = module.vpc.network_id
  db_password    = var.db_password
  vpc_dependency = module.vpc.vpc_connection
}

# ==========================================
# Cloud Storage Submodule
# ==========================================
module "cloud_storage" {
  source      = "./modules/cloud_storage"
  app_name    = var.app_name
  environment = var.environment
  region      = var.region
}

# ==========================================
# Artifact Registry Submodule
# ==========================================
module "artifact_registry" {
  source      = "./modules/artifact_registry"
  app_name    = var.app_name
  environment = var.environment
  region      = var.region
}

# ==========================================
# Secret Manager Submodule
# ==========================================
module "secrets" {
  source      = "./modules/secrets"
  app_name    = var.app_name
  environment = var.environment
}

# ==========================================
# Cloud Run Service: FastAPI Backend
# ==========================================
module "cloud_run_backend" {
  source         = "./modules/cloud_run"
  app_name       = var.app_name
  environment    = var.environment
  region         = var.region
  service_name   = "backend"
  image          = "${module.artifact_registry.repository_url}/backend:latest"
  container_port = 8000

  vpc_connector_id = module.vpc.vpc_connector_id

  env_vars = {
    "APP_ENV"             = "production"
    "QDRANT_URL"          = "http://qdrant-shared-host:6333" # Qdrant SaaS or shared host
    "GCS_BUCKET_NAME"     = module.cloud_storage.document_bucket_name
    "GCS_BUCKET_PROCESSED" = module.cloud_storage.processed_bucket_name
    "GUARDRAILS_ENABLED"  = "true"
  }

  secret_env_vars = {
    "PORTKEY_API_KEY" = { secret_id = module.secrets.portkey_secret_id }
    "POSTGRES_URL"    = { secret_id = module.secrets.db_url_secret_id }
  }
}

# ==========================================
# Cloud Run Service: Streamlit Chat
# ==========================================
module "cloud_run_frontend_chat" {
  source         = "./modules/cloud_run"
  app_name       = var.app_name
  environment    = var.environment
  region         = var.region
  service_name   = "chat-ui"
  image          = "${module.artifact_registry.repository_url}/frontend:latest"
  container_port = 8501

  env_vars = {
    "BACKEND_URL" = "${module.cloud_run_backend.service_url}/api/v1"
    "API_KEY"     = "changeme"
  }
}

# ==========================================
# Cloud Run Service: Streamlit Evaluation Dashboard
# ==========================================
module "cloud_run_frontend_eval" {
  source         = "./modules/cloud_run"
  app_name       = var.app_name
  environment    = var.environment
  region         = var.region
  service_name   = "eval-ui"
  image          = "${module.artifact_registry.repository_url}/frontend:latest"
  container_port = 8502

  # Pass custom entrypoint arguments to launch eval app
  container_args = ["run", "frontend/eval_app.py", "--server.port=8502"]

  env_vars = {
    "BACKEND_URL" = "${module.cloud_run_backend.service_url}/api/v1"
    "API_KEY"     = "changeme"
  }
}
