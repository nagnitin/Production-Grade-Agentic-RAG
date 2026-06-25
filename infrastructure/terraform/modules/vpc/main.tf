# ==========================================
# VPC Network & Subnet
# ==========================================
resource "google_compute_network" "vpc_network" {
  name                    = "${var.app_name}-${var.environment}-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet" {
  name          = "${var.app_name}-${var.environment}-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc_network.id
}

# ==========================================
# Serverless VPC Access Connector (Cloud Run Private Access)
# ==========================================
resource "google_vpc_access_connector" "connector" {
  name          = "${var.app_name}-vpc-conn"
  region        = var.region
  ip_cidr_range = "10.8.0.0/28"
  network       = google_compute_network.vpc_network.name
}

# ==========================================
# Private IP Allocation for Cloud SQL Service Peering
# ==========================================
resource "google_compute_global_address" "private_ip_alloc" {
  name          = "${var.app_name}-${var.environment}-sql-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc_network.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc_network.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_alloc.name]
}

# Variables & Outputs definitions inline for module encapsulation
variable "app_name" {}
variable "environment" {}
variable "region" {}

output "network_id" {
  value = google_compute_network.vpc_network.id
}
output "vpc_connector_id" {
  value = google_vpc_access_connector.connector.id
}
output "vpc_connection" {
  value = google_service_networking_connection.private_vpc_connection
}
