// Package client provides a gRPC client for the Python inference service.
package client

import (
	"context"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"

	pb "github.com/iris3d/go/proto"
)

// InferenceClient wraps the gRPC client for the Python inference service.
type InferenceClient struct {
	conn   *grpc.ClientConn
	client pb.InferenceServiceClient
	mu     sync.RWMutex
}

// Config holds client configuration options.
type Config struct {
	// Address of the inference service (host:port)
	Address string

	// Connection timeout
	ConnectTimeout time.Duration

	// Request timeout
	RequestTimeout time.Duration

	// Maximum message size in bytes
	MaxMessageSize int

	// Enable connection keepalive
	EnableKeepalive bool
}

// DefaultConfig returns sensible default configuration.
func DefaultConfig(address string) Config {
	return Config{
		Address:         address,
		ConnectTimeout:  10 * time.Second,
		RequestTimeout:  60 * time.Second,
		MaxMessageSize:  100 * 1024 * 1024, // 100MB
		EnableKeepalive: true,
	}
}

// NewInferenceClient creates a new client connected to the inference service.
func NewInferenceClient(cfg Config) (*InferenceClient, error) {
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(cfg.MaxMessageSize),
			grpc.MaxCallSendMsgSize(cfg.MaxMessageSize),
		),
	}

	if cfg.EnableKeepalive {
		opts = append(opts, grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                10 * time.Second,
			Timeout:             3 * time.Second,
			PermitWithoutStream: true,
		}))
	}

	ctx, cancel := context.WithTimeout(context.Background(), cfg.ConnectTimeout)
	defer cancel()

	conn, err := grpc.DialContext(ctx, cfg.Address, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to inference service: %w", err)
	}

	return &InferenceClient{
		conn:   conn,
		client: pb.NewInferenceServiceClient(conn),
	}, nil
}

// Close closes the client connection.
func (c *InferenceClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// Infer runs depth estimation and 3D projection on an image.
func (c *InferenceClient) Infer(
	ctx context.Context,
	imageData []byte,
	format string,
	width, height int32,
	intrinsics *pb.CameraIntrinsics,
) (*pb.InferenceResponse, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	req := &pb.InferenceRequest{
		ImageData:  imageData,
		Format:     format,
		Width:      width,
		Height:     height,
		Intrinsics: intrinsics,
	}

	return c.client.Infer(ctx, req)
}

// InferImage is a convenience method for processing an encoded image.
func (c *InferenceClient) InferImage(
	ctx context.Context,
	imageData []byte,
	format string,
) (*pb.InferenceResponse, error) {
	return c.Infer(ctx, imageData, format, 0, 0, nil)
}

// InferRaw processes a raw RGB image.
func (c *InferenceClient) InferRaw(
	ctx context.Context,
	imageData []byte,
	width, height int32,
) (*pb.InferenceResponse, error) {
	return c.Infer(ctx, imageData, "raw", width, height, nil)
}

// InferWithIntrinsics processes an image with custom camera intrinsics.
func (c *InferenceClient) InferWithIntrinsics(
	ctx context.Context,
	imageData []byte,
	format string,
	fx, fy, cx, cy float32,
) (*pb.InferenceResponse, error) {
	intrinsics := &pb.CameraIntrinsics{
		FocalLengthX:    fx,
		FocalLengthY:    fy,
		PrincipalPointX: cx,
		PrincipalPointY: cy,
	}
	return c.Infer(ctx, imageData, format, 0, 0, intrinsics)
}

// Pool manages a pool of inference clients for concurrent access.
type Pool struct {
	clients []*InferenceClient
	index   int
	mu      sync.Mutex
}

// NewPool creates a pool of inference clients.
func NewPool(cfg Config, size int) (*Pool, error) {
	if size < 1 {
		size = 1
	}

	clients := make([]*InferenceClient, size)
	for i := 0; i < size; i++ {
		client, err := NewInferenceClient(cfg)
		if err != nil {
			// Close any clients we've already created
			for j := 0; j < i; j++ {
				clients[j].Close()
			}
			return nil, fmt.Errorf("failed to create client %d: %w", i, err)
		}
		clients[i] = client
	}

	return &Pool{clients: clients}, nil
}

// Get returns the next available client in round-robin fashion.
func (p *Pool) Get() *InferenceClient {
	p.mu.Lock()
	defer p.mu.Unlock()

	client := p.clients[p.index]
	p.index = (p.index + 1) % len(p.clients)
	return client
}

// Close closes all clients in the pool.
func (p *Pool) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	var lastErr error
	for _, client := range p.clients {
		if err := client.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}
