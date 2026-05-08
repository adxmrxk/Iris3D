# Iris3D Go Gateway Service
# Multi-stage build for minimal production image

# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /build

# Install protoc and plugins
RUN apk add --no-cache protobuf protobuf-dev

# Copy go mod files first for better caching
COPY go.mod go.sum ./
RUN go mod download

# Copy proto files and generate
COPY proto/ ./proto/
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest && \
    mkdir -p proto && \
    protoc --go_out=. --go-grpc_out=. proto/iris3d.proto

# Copy source code
COPY cmd/ ./cmd/
COPY internal/ ./internal/

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o iris3d-gateway ./cmd/server

# Production stage
FROM alpine:3.18

WORKDIR /app

# Add CA certificates for HTTPS calls if needed
RUN apk --no-cache add ca-certificates

# Copy binary from builder
COPY --from=builder /build/iris3d-gateway .

# Expose gRPC port
EXPOSE 50051

# Health check using grpc_health_probe
# RUN wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/v0.4.22/grpc_health_probe-linux-amd64 && \
#     chmod +x /bin/grpc_health_probe
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD ["/bin/grpc_health_probe", "-addr=:50051"]

# Run the gateway
ENTRYPOINT ["./iris3d-gateway"]
CMD ["--port", "50051"]
