variable "project_id" {
  type        = string
  description = "The Google Cloud Project ID"
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "The region to deploy resources in"
}

variable "environment" {
  type        = string
  default     = "dev"
  description = "Deployment environment (e.g. dev, staging, prod)"
}

variable "app_name" {
  type        = string
  default     = "agentic-rag"
  description = "The application name prefix used in resources"
}

variable "db_password" {
  type        = string
  sensitive   = true
  description = "The master password for the Cloud SQL PostgreSQL instance"
}
