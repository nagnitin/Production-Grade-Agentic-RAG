resource "google_sql_database_instance" "db_instance" {
  name             = "${var.app_name}-${var.environment}-db"
  database_version = "POSTGRES_16"
  region           = var.region

  # Ensure peering is created before DB instance
  depends_on = [var.vpc_dependency]

  settings {
    tier = "db-f1-micro" # Production should use db-custom class
    ip_configuration {
      ipv4_enabled    = false
      private_network = var.network_id
    }
    backup_configuration {
      enabled = true
    }
  }

  deletion_protection = false # True for production deployment
}

resource "google_sql_database" "database" {
  name     = "rag"
  instance = google_sql_database_instance.db_instance.name
}

resource "google_sql_user" "db_user" {
  name     = "postgres"
  instance = google_sql_database_instance.db_instance.name
  password = var.db_password
}

# Module inputs/outputs
variable "app_name" {}
variable "environment" {}
variable "region" {}
variable "network_id" {}
variable "db_password" {}
variable "vpc_dependency" {
  description = "Peering connection dependency to prevent race conditions during DB provision"
}

output "connection_name" {
  value = google_sql_database_instance.db_instance.connection_name
}
output "db_name" {
  value = google_sql_database.database.name
}
output "db_user" {
  value = google_sql_user.db_user.name
}
output "db_ip" {
  value = google_sql_database_instance.db_instance.private_ip_address
}
