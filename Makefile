# Iris3D v2.0 Makefile
# Build automation for the polyglot 3D reconstruction service

.PHONY: all build proto clean test docker help

# ============================================================================
# Variables
# ============================================================================

GO_MODULE := github.com/iris3d/go
PYTHON_DIR := python
GO_DIR := go
WEB_DIR := web
PROTO_DIR := proto

# ============================================================================
# Default Target
# ============================================================================

all: proto build

# ============================================================================
# Protocol Buffer Generation (using Buf)
# ============================================================================

proto: proto-lint proto-generate
	@echo "Protocol buffers generated successfully"

proto-lint:
	@echo "Linting protos..."
	buf lint $(PROTO_DIR)/

proto-generate:
	@echo "Generating code from protos..."
	buf generate

proto-legacy: python-proto go-proto
	@echo "Legacy proto generation complete"

python-proto:
	@echo "Generating Python protobuf files..."
	python -m grpc_tools.protoc \
		-I$(PROTO_DIR) \
		--python_out=$(PYTHON_DIR)/gen \
		--grpc_python_out=$(PYTHON_DIR)/gen \
		$(PROTO_DIR)/iris3d/v1/*.proto

go-proto:
	@echo "Generating Go protobuf files..."
	mkdir -p $(GO_DIR)/gen/iris3d/v1
	protoc \
		--go_out=$(GO_DIR)/gen \
		--go_opt=paths=source_relative \
		--go-grpc_out=$(GO_DIR)/gen \
		--go-grpc_opt=paths=source_relative \
		--connect-go_out=$(GO_DIR)/gen \
		--connect-go_opt=paths=source_relative \
		-I$(PROTO_DIR) \
		$(PROTO_DIR)/iris3d/v1/*.proto

# ============================================================================
# Model Export
# ============================================================================

export-models: export-midas export-depth-anything
	@echo "All models exported"

export-midas:
	@echo "Exporting MiDaS model to ONNX..."
	cd $(PYTHON_DIR) && python export_onnx.py --output ../models/midas_v31_dpt_large.onnx

export-depth-anything:
	@echo "Note: Depth Anything v2 is loaded from HuggingFace at runtime"

# ============================================================================
# Build
# ============================================================================

build: build-go build-web
	@echo "Build complete"

build-go:
	@echo "Building Go gateway..."
	cd $(GO_DIR) && go build -o ../bin/iris3d-gateway ./cmd/server
	@echo "Binary created: ./bin/iris3d-gateway"

build-go-linux:
	@echo "Building Go gateway for Linux..."
	cd $(GO_DIR) && GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -ldflags="-w -s" -o ../bin/iris3d-gateway-linux ./cmd/server

build-web:
	@echo "Building web viewer..."
	cd $(WEB_DIR) && npm ci && npm run build

# ============================================================================
# Docker
# ============================================================================

docker: docker-build docker-up

docker-build:
	@echo "Building Docker images..."
	docker-compose -f deploy/docker-compose.yaml build

docker-up:
	@echo "Starting services..."
	docker-compose -f deploy/docker-compose.yaml up -d
	@echo "Services started:"
	@echo "  Gateway:    http://localhost:8080 (Connect) / localhost:50051 (gRPC)"
	@echo "  Viewer:     http://localhost:3000"
	@echo "  Jaeger:     http://localhost:16686"
	@echo "  Prometheus: http://localhost:9091"
	@echo "  Grafana:    http://localhost:3001 (admin/iris3d)"

docker-down:
	@echo "Stopping services..."
	docker-compose -f deploy/docker-compose.yaml down

docker-logs:
	docker-compose -f deploy/docker-compose.yaml logs -f

docker-clean:
	docker-compose -f deploy/docker-compose.yaml down -v --rmi local

# ============================================================================
# Kubernetes
# ============================================================================

k8s-deploy-dev:
	@echo "Deploying to dev namespace..."
	kubectl apply -k deploy/kubernetes/overlays/dev

k8s-deploy-prod:
	@echo "Deploying to production namespace..."
	kubectl apply -k deploy/kubernetes/overlays/prod

k8s-delete:
	kubectl delete -k deploy/kubernetes/base

# ============================================================================
# Development
# ============================================================================

install: install-go install-python install-web install-tools
	@echo "All dependencies installed"

install-go:
	cd $(GO_DIR) && go mod download

install-python:
	cd $(PYTHON_DIR) && pip install -r requirements.txt

install-web:
	cd $(WEB_DIR) && npm ci

install-tools:
	@echo "Installing development tools..."
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
	go install connectrpc.com/connect/cmd/protoc-gen-connect-go@latest
	go install github.com/bufbuild/buf/cmd/buf@latest
	pip install grpcio-tools ruff pytest pytest-cov

# Run services locally
run-inference:
	cd $(PYTHON_DIR) && python inference_server.py --config config.yaml

run-gateway:
	cd $(GO_DIR) && go run ./cmd/server --inference-host localhost:50052

run-web:
	cd $(WEB_DIR) && npm run dev

# ============================================================================
# Testing
# ============================================================================

test: test-go test-python
	@echo "All tests passed"

test-go:
	@echo "Running Go tests..."
	cd $(GO_DIR) && go test -v -race -cover ./...

test-python:
	@echo "Running Python tests..."
	cd $(PYTHON_DIR) && pytest -v --cov=. --cov-report=term-missing

test-integration:
	@echo "Running integration tests..."
	./scripts/integration-test.sh

# ============================================================================
# Linting
# ============================================================================

lint: lint-proto lint-go lint-python
	@echo "Linting complete"

lint-proto:
	buf lint $(PROTO_DIR)/

lint-go:
	cd $(GO_DIR) && golangci-lint run

lint-python:
	cd $(PYTHON_DIR) && ruff check .

fmt: fmt-go fmt-python
	@echo "Formatting complete"

fmt-go:
	cd $(GO_DIR) && go fmt ./...

fmt-python:
	cd $(PYTHON_DIR) && ruff format .

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "Cleaning build artifacts..."
	rm -rf bin/
	rm -rf $(GO_DIR)/gen/
	rm -rf $(PYTHON_DIR)/gen/
	rm -rf $(PYTHON_DIR)/__pycache__
	rm -rf $(PYTHON_DIR)/**/__pycache__
	rm -rf $(WEB_DIR)/dist
	rm -rf $(WEB_DIR)/node_modules
	@echo "Clean complete"

# ============================================================================
# Help
# ============================================================================

help:
	@echo "Iris3D v2.0 Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Main Targets:"
	@echo "  all              Build everything (default)"
	@echo "  proto            Generate protobuf code with Buf"
	@echo "  build            Build Go binary and web viewer"
	@echo "  docker           Build and start Docker containers"
	@echo "  test             Run all tests"
	@echo "  lint             Run all linters"
	@echo ""
	@echo "Development:"
	@echo "  install          Install all dependencies"
	@echo "  run-inference    Run Python inference service"
	@echo "  run-gateway      Run Go gateway service"
	@echo "  run-web          Run web viewer dev server"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker images"
	@echo "  docker-up        Start Docker containers"
	@echo "  docker-down      Stop Docker containers"
	@echo "  docker-logs      Follow Docker logs"
	@echo ""
	@echo "Kubernetes:"
	@echo "  k8s-deploy-dev   Deploy to dev namespace"
	@echo "  k8s-deploy-prod  Deploy to production"
	@echo ""
	@echo "Model Export:"
	@echo "  export-models    Export all depth models to ONNX"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            Remove build artifacts"
	@echo "  docker-clean     Remove Docker containers and images"
