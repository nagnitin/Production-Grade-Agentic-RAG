output "backend_service_url" {
  value       = module.cloud_run_backend.service_url
  description = "The URL of the deployed FastAPI backend service"
}

output "chat_frontend_url" {
  value       = module.cloud_run_frontend_chat.service_url
  description = "The URL of the deployed Streamlit Chat frontend service"
}

output "eval_frontend_url" {
  value       = module.cloud_run_frontend_eval.service_url
  description = "The URL of the deployed Streamlit Evaluation dashboard service"
}

output "artifact_registry_repository" {
  value       = module.artifact_registry.repository_url
  description = "The URL of the docker registry repository"
}

output "db_instance_connection_name" {
  value       = module.cloud_sql.connection_name
  description = "The connection name of the PostgreSQL instance"
}

output "document_bucket_name" {
  value       = module.cloud_storage.document_bucket_name
  description = "The name of the GCS bucket for raw document uploads"
}
