# Makefile for superconductor discovery platform

.PHONY: help build up down test clean logs shell migrate

# Default target
help:
	@echo "Superconductor Discovery Platform - Development Commands"
	@echo "======================================================"
	@echo "  make build          - Build all Docker containers"
	@echo "  make up             - Start development environment"
	@echo "  make down           - Stop all containers"
	@echo "  make test           - Run test suite"
	@echo "  make clean          - Clean up containers and volumes"
	@echo "  make logs           - Show container logs"
	@echo "  make shell          - Open shell in API container"
	@echo "  make migrate        - Run database migrations"
	@echo "  make psql           - Open PostgreSQL shell"
	@echo "  make redis-cli      - Open Redis CLI"
	@echo "  make format         - Format code with black/isort"
	@echo "  make lint           - Run linting checks"
	@echo "  make security       - Run security checks"
	@echo "  make monitoring     - Open monitoring dashboards"

# Build all containers
build:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml build

# Start development environment
up:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
	@echo "Services starting..."
	@echo "API: http://localhost:8000"
	@echo "Grafana: http://localhost:3000 (admin/admin123)"
	@echo "Flower: http://localhost:5555"
	@echo "pgAdmin: http://localhost:5050"
	@echo "MinIO: http://localhost:9001 (minioadmin/minioadmin123)"
	@echo "Jupyter: http://localhost:8888"
	@echo "MailHog: http://localhost:8025"

# Stop all containers
down:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml down

# Run tests
test:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm test-runner

# Run specific test file
test-file:
	@read -p "Test file path: " testfile; \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm test-runner pytest $$testfile -v

# Clean up everything
clean:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml down -v
	docker system prune -f
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Show logs
logs:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

# Show logs for specific service
logs-service:
	@read -p "Service name: " service; \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f $$service

# Open shell in API container
shell:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec api bash

# Run database migrations
migrate:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm flyway migrate

# Create new migration
migration-create:
	@read -p "Migration name: " name; \
	timestamp=$$(date +%Y%m%d%H%M%S); \
	touch "db/migration/V$${timestamp}__$${name}.sql"; \
	echo "Created: db/migration/V$${timestamp}__$${name}.sql"

# Open PostgreSQL shell
psql:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec postgres psql -U postgres -d superconductor_discovery

# Open Redis CLI
redis-cli:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec redis redis-cli

# Format code
format:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm dev-tools black /app
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm dev-tools isort /app

# Run linting
lint:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm dev-tools flake8 /app
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm dev-tools mypy /app

# Run security checks
security:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm dev-tools bandit -r /app

# Open monitoring dashboards
monitoring:
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000 (admin/admin123)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Flower (Celery): http://localhost:5555"

# Check service health
health:
	@echo "Checking service health..."
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps
	@echo "\nDatabase:"
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec postgres pg_isready -U postgres || echo "Database not ready"
	@echo "\nRedis:"
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec redis redis-cli ping || echo "Redis not ready"
	@echo "\nMinIO:"
	@curl -s http://localhost:9000/minio/health/live || echo "MinIO not ready"

# Initialize development environment
init: build
	cp .env.example .env
	@echo "Environment file created. Please edit .env with your configuration."
	$(MAKE) up
	@echo "Waiting for services to start..."
	@sleep 10
	$(MAKE) migrate
	@echo "Development environment ready!"

# Run a quantum simulation test
test-simulation:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec api python -m tests.test_integration_simulation

# Backup database
backup-db:
	@mkdir -p backups
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec postgres pg_dump -U postgres superconductor_discovery > "backups/backup_$$timestamp.sql"
	@echo "Database backed up to backups/backup_$$timestamp.sql"

# Restore database
restore-db:
	@read -p "Backup file path: " backup; \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec -T postgres psql -U postgres superconductor_discovery < $$backup

# GPU check
gpu-check:
	@echo "Checking GPU availability..."
	@docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi