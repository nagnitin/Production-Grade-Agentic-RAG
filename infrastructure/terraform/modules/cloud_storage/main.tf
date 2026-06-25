resource "google_storage_bucket" "raw_bucket" {
  name          = "${var.app_name}-${var.environment}-raw-documents"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true
}

resource "google_storage_bucket" "processed_bucket" {
  name          = "${var.app_name}-${var.environment}-processed-documents"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true
}

variable "app_name" {}
variable "environment" {}
variable "region" {}

output "document_bucket_name" {
  value = google_storage_bucket.raw_bucket.name
}
output "processed_bucket_name" {
  value = google_storage_bucket.processed_bucket.name
}
