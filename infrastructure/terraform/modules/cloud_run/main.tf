resource "google_cloud_run_v2_service" "service" {
  name     = "${var.app_name}-${var.environment}-${var.service_name}"
  location = var.region

  template {
    containers {
      image = var.image
      args  = var.container_args

      ports {
        container_port = var.container_port
      }

      # Inject standard environment variables
      dynamic "env" {
        for_each = var.env_vars
        content {
          name  = env.key
          value = env.value
        }
      }

      # Inject Secret Manager secrets
      dynamic "env" {
        for_each = var.secret_env_vars
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value.secret_id
              version = "latest"
            }
          }
        }
      }
    }

    # Bind to VPC access connector if provided
    dynamic "vpc_access" {
      for_each = var.vpc_connector_id != "" ? [1] : []
      content {
        connector = var.vpc_connector_id
        egress    = "ALL_TRAFFIC"
      }
    }
  }
}

# Grant public unauthenticated access if configured
resource "google_cloud_run_v2_service_iam_member" "noauth" {
  count    = var.allow_public ? 1 : 0
  location = google_cloud_run_v2_service.service.location
  name     = google_cloud_run_v2_service.service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Inputs/Outputs
variable "app_name" {}
variable "environment" {}
variable "region" {}
variable "service_name" {}
variable "image" {}
variable "container_port" {
  type    = number
  default = 8000
}
variable "env_vars" {
  type    = map(string)
  default = {}
}
variable "secret_env_vars" {
  type    = map(any)
  default = {}
}
variable "vpc_connector_id" {
  type    = string
  default = ""
}
variable "allow_public" {
  type    = bool
  default = true
}
variable "container_args" {
  type    = list(string)
  default = []
}

output "service_url" {
  value = google_cloud_run_v2_service.service.uri
}
