resource "google_secret_manager_secret" "portkey_key" {
  secret_id = "${var.app_name}-${var.environment}-portkey-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "db_url" {
  secret_id = "${var.app_name}-${var.environment}-db-url"

  replication {
    auto {}
  }
}

variable "app_name" {}
variable "environment" {}

output "portkey_secret_id" {
  value = google_secret_manager_secret.portkey_key.secret_id
}
output "db_url_secret_id" {
  value = google_secret_manager_secret.db_url.secret_id
}
