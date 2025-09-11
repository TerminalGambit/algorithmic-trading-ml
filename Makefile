# Algorithmic Trading ML - Development Makefile

.PHONY: help install dev-install test lint format clean docker-build docker-up docker-down setup-dev

# Ensure poetry is in PATH
PATH := $(HOME)/.local/bin:$(PATH)
export PATH

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
NC = \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Algorithmic Trading ML - Development Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Install production dependencies
	poetry install --no-dev

dev-install: ## Install all dependencies including dev tools
	@if ! command -v poetry > /dev/null; then \
		echo "$(YELLOW)Poetry not found. Attempting to install poetry...$(NC)"; \
		python3 -m pip install poetry; \
	fi
	poetry run install
	poetry run pre-commit install

setup-dev: ## Complete development environment setup
	@echo "$(GREEN)Setting up development environment...$(NC)"
	cp .env.template .env
	@echo "$(YELLOW)Please edit .env file with your API keys$(NC)"
	make dev-install
	make docker-up
	@echo "$(GREEN)Development environment ready!$(NC)"

test: ## Run all tests
	poetry run pytest

test-unit: ## Run unit tests only
	poetry run pytest tests/unit -v

test-integration: ## Run integration tests only
	poetry run pytest tests/integration -v

test-e2e: ## Run end-to-end tests only
	poetry run pytest tests/e2e -v

test-coverage: ## Run tests with coverage report
	poetry run pytest --cov=src --cov-report=html --cov-report=term-missing

lint: ## Run all linting tools
	poetry run black --check .
	poetry run isort --check-only .
	poetry run flake8 .
	poetry run mypy src/
	poetry run bandit -r src/

format: ## Format code with black and isort
	poetry run black .
	poetry run isort .

security: ## Run security analysis
	poetry run bandit -r src/

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/ dist/ build/

docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start all services with Docker Compose
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-clean: ## Clean up Docker containers and volumes
	docker-compose down -v
	docker system prune -f

# Data collection commands
collect-data: ## Run data collection pipeline
	poetry run python -m src.data_collection.main

# Model training commands
train-model: ## Train ML models
	poetry run python -m src.models.train

# Backtesting commands
run-backtest: ## Run backtesting pipeline
	poetry run python -m src.backtesting.main

# MLflow commands
mlflow-ui: ## Start MLflow UI (if not using Docker)
	poetry run mlflow ui --host 0.0.0.0 --port 5000

# Database commands
db-migrate: ## Run database migrations
	poetry run alembic upgrade head

db-revision: ## Create new database revision
	@read -p "Enter migration message: " msg; \
	poetry run alembic revision --autogenerate -m "$$msg"

# Jupyter commands
jupyter: ## Start Jupyter Lab
	poetry run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Performance testing
benchmark: ## Run performance benchmarks
	poetry run pytest tests/ --benchmark-only

# Quality checks
check-all: lint test ## Run all quality checks

# Production deployment (placeholder)
deploy-staging: ## Deploy to staging environment
	@echo "$(YELLOW)Staging deployment not implemented yet$(NC)"

deploy-prod: ## Deploy to production environment
	@echo "$(YELLOW)Production deployment not implemented yet$(NC)"
