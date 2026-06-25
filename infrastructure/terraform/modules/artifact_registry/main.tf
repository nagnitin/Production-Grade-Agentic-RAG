resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "${var.app_name}-${var.environment}-repo"
  description   = "Docker repository for Agentic RAG services"
  format        = "DOCKER"
}

variable "app_name" {}
variable "environment" {}
variable "region" {}

output "repository_url" {
  value = "${google_artifact_registry_repository.repo.location}-docker.pkg.dev/${google_artifact_registry_repository.repo.project}/${google_artifact_registry_repository.repo.repository_id}"
}
