// Package main provides the entry point for the Iris3D Go gateway service.
package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	grpcserver "github.com/iris3d/go/internal/grpc"
	pb "github.com/iris3d/go/proto"
)

func main() {
	// Parse command line flags
	port := flag.Int("port", 50051, "gRPC server port")
	inferenceHost := flag.String("inference-host", "localhost:50052", "Inference service address")
	voxelSize := flag.Float64("voxel-size", 0.01, "Default voxel size for downsampling")
	flag.Parse()

	// Allow environment variable overrides
	if envHost := os.Getenv("INFERENCE_HOST"); envHost != "" {
		*inferenceHost = envHost
	}
	if envPort := os.Getenv("PORT"); envPort != "" {
		fmt.Sscanf(envPort, "%d", port)
	}

	log.Printf("Iris3D Gateway starting...")
	log.Printf("  Listen port: %d", *port)
	log.Printf("  Inference service: %s", *inferenceHost)
	log.Printf("  Default voxel size: %.4f", *voxelSize)

	// Create gRPC server
	serverCfg := grpcserver.Config{
		InferenceAddress: *inferenceHost,
		VoxelSize:        float32(*voxelSize),
	}

	iris3dServer, err := grpcserver.NewServer(serverCfg)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}
	defer iris3dServer.Close()

	// Create gRPC server with options
	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(100*1024*1024), // 100MB
		grpc.MaxSendMsgSize(100*1024*1024),
	)

	// Register services
	pb.RegisterIris3DServer(grpcServer, iris3dServer)

	// Enable reflection for debugging with grpcurl
	reflection.Register(grpcServer)

	// Start listening
	address := fmt.Sprintf(":%d", *port)
	listener, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigChan
		log.Printf("Received signal %v, shutting down...", sig)
		grpcServer.GracefulStop()
	}()

	log.Printf("Iris3D Gateway listening on %s", address)

	if err := grpcServer.Serve(listener); err != nil {
		log.Fatalf("Server error: %v", err)
	}

	log.Println("Server stopped")
}
