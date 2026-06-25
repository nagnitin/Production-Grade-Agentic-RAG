.PHONY: help install install-dev lint format test test-unit test-integration test-e2e \
       docker-up docker-down docker-build api frontend migrate seed clean deploy

# ============================================================
# Agentic RAG Platform — Makefile
# ============================================================

PYTHON := python3
PIP := pip
DOCKER_COMPOSE := docker compose

# Colors
GREEN  := \033[0;32m
YELLOW := \033[0;33m
CYAN   := \033[0;36m
RESET  := \033[0m

help: ## Show this help message
	@echo "$(CYAN)Agentic RAG Platform$(RESET)"
	@echo "$(YELLOW)Usage:$(RESET) make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

# === Setup ===

install: ## Install production dependencies
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev,frontend]"
	pre-commit install

# === Code Quality ===

lint: ## Run linting (ruff + mypy)
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format: ## Auto-format code
	ruff check --fix src/ tests/
	ruff format src/ tests/

# === Testing ===

test: ## Run all tests
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	pytest tests/unit/ -v -m unit

test-integration: ## Run integration tests (requires Docker services)
	pytest tests/integration/ -v -m integration

test-e2e: ## Run end-to-end tests
	pytest tests/e2e/ -v -m e2e

# === Local Development ===

api: ## Run FastAPI server locally
	uvicorn src.api.main:create_app --factory --host 0.0.0.0 --port 8000 --reload

frontend: ## Run Streamlit chat app locally
	streamlit run frontend/app.py --server.port 8501

eval-app: ## Run Streamlit evaluation app locally
	streamlit run frontend/eval_app.py --server.port 8502

# === Docker ===

docker-up: ## Start all services with Docker Compose
	$(DOCKER_COMPOSE) up -d

docker-down: ## Stop all Docker services
	$(DOCKER_COMPOSE) down

docker-build: ## Build all Docker images
	$(DOCKER_COMPOSE) build

docker-logs: ## Tail Docker logs
	$(DOCKER_COMPOSE) logs -f

docker-ps: ## Show running containers
	$(DOCKER_COMPOSE) ps

# === Database ===

migrate: ## Run database migrations
	alembic upgrade head

migrate-create: ## Create new migration (usage: make migrate-create MSG="description")
	alembic revision --autogenerate -m "$(MSG)"

migrate-rollback: ## Rollback last migration
	alembic downgrade -1

seed: ## Seed golden evaluation dataset
	$(PYTHON) -m src.evaluation.datasets.golden_dataset seed

# === Deployment ===

deploy-dev: ## Deploy to GCP dev environment
	./scripts/deploy.sh dev

deploy-staging: ## Deploy to GCP staging environment
	./scripts/deploy.sh staging

deploy-prod: ## Deploy to GCP production environment
	./scripts/deploy.sh prod

terraform-plan: ## Run Terraform plan
	cd infrastructure/terraform && terraform plan -var-file=environments/dev.tfvars

terraform-apply: ## Apply Terraform changes
	cd infrastructure/terraform && terraform apply -var-file=environments/dev.tfvars

# === Evaluation ===

evaluate: ## Run RAGAS evaluation
	$(PYTHON) -m src.evaluation.ragas_evaluator

# === Cleanup ===

clean: ## Clean build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/
