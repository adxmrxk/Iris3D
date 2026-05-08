// Package grpc provides the public-facing gRPC API gateway for Iris3D.
package grpc

import (
	"context"
	"fmt"
	"io"
	"log"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/iris3d/go/internal/client"
	"github.com/iris3d/go/internal/ply"
	"github.com/iris3d/go/internal/voxel"
	pb "github.com/iris3d/go/proto"
)

// Server implements the Iris3D gRPC service.
type Server struct {
	pb.UnimplementedIris3DServer

	inferenceClient *client.InferenceClient
	voxelConfig     voxel.Config
	requestTimeout  time.Duration
}

// Config holds server configuration.
type Config struct {
	// InferenceAddress is the address of the Python inference service
	InferenceAddress string

	// VoxelSize for default downsampling
	VoxelSize float32

	// RequestTimeout for processing
	RequestTimeout time.Duration
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		InferenceAddress: "localhost:50052",
		VoxelSize:        0.01,
		RequestTimeout:   60 * time.Second,
	}
}

// NewServer creates a new Iris3D gRPC server.
func NewServer(cfg Config) (*Server, error) {
	// Create inference client
	clientCfg := client.DefaultConfig(cfg.InferenceAddress)
	inferenceClient, err := client.NewInferenceClient(clientCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create inference client: %w", err)
	}

	return &Server{
		inferenceClient: inferenceClient,
		voxelConfig: voxel.Config{
			VoxelSize:   cfg.VoxelSize,
			UseCentroid: false,
		},
		requestTimeout: cfg.RequestTimeout,
	}, nil
}

// Close cleans up server resources.
func (s *Server) Close() error {
	return s.inferenceClient.Close()
}

// ProcessImage processes a single image and returns a point cloud.
func (s *Server) ProcessImage(
	ctx context.Context,
	req *pb.ImageRequest,
) (*pb.PointCloudResponse, error) {
	startTime := time.Now()

	// Validate request
	if len(req.ImageData) == 0 {
		return nil, status.Error(codes.InvalidArgument, "image_data is required")
	}

	// Set timeout
	ctx, cancel := context.WithTimeout(ctx, s.requestTimeout)
	defer cancel()

	// Call inference service
	inferResp, err := s.inferenceClient.Infer(
		ctx,
		req.ImageData,
		req.Format,
		req.Width,
		req.Height,
		req.Intrinsics,
	)
	if err != nil {
		log.Printf("Inference error: %v", err)
		return nil, status.Errorf(codes.Internal, "inference failed: %v", err)
	}

	// Convert raw point cloud to PLY points
	points, err := ply.PointsFromRaw(inferResp.PointCloudData, int(inferResp.NumPoints))
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to parse point cloud: %v", err)
	}

	originalCount := len(points)

	// Apply voxel downsampling if enabled
	var downsampleTimeMs float64
	if req.Downsampling != nil && req.Downsampling.Enabled {
		voxelSize := req.Downsampling.VoxelSize
		if voxelSize <= 0 {
			voxelSize = s.voxelConfig.VoxelSize
		}

		downsampler := voxel.NewWithSize(voxelSize)
		points, downsampleTimeMs = downsampler.Downsample(points)
	}

	// Write PLY file
	writer := ply.NewBinaryWriter()
	plyData, err := writer.WriteToBytes(points)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to write PLY: %v", err)
	}

	totalTimeMs := float32(time.Since(startTime).Milliseconds())

	log.Printf(
		"ProcessImage: %d -> %d points, inference=%.1fms, projection=%.1fms, downsample=%.1fms, total=%.1fms",
		originalCount, len(points),
		inferResp.InferenceTimeMs, inferResp.ProjectionTimeMs,
		downsampleTimeMs, totalTimeMs,
	)

	return &pb.PointCloudResponse{
		PlyData:        plyData,
		NumPoints:      int32(len(points)),
		OriginalPoints: int32(originalCount),
		Stats: &pb.ProcessingStats{
			InferenceTimeMs:    inferResp.InferenceTimeMs,
			ProjectionTimeMs:   inferResp.ProjectionTimeMs,
			DownsamplingTimeMs: float32(downsampleTimeMs),
			TotalTimeMs:        totalTimeMs,
		},
	}, nil
}

// ProcessVideoStream processes a stream of video frames.
func (s *Server) ProcessVideoStream(
	stream pb.Iris3D_ProcessVideoStreamServer,
) error {
	log.Println("Video stream started")

	var (
		frameCount int
		wg         sync.WaitGroup
		mu         sync.Mutex
		streamErr  error
	)

	// Process frames as they arrive
	for {
		frame, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Printf("Stream receive error: %v", err)
			return status.Errorf(codes.Internal, "stream error: %v", err)
		}

		frameCount++
		frameID := frame.FrameId

		// Process frame in goroutine for pipelining
		wg.Add(1)
		go func(f *pb.VideoFrame) {
			defer wg.Done()

			resp, err := s.processFrame(stream.Context(), f)
			if err != nil {
				log.Printf("Frame %d error: %v", frameID, err)
				mu.Lock()
				if streamErr == nil {
					streamErr = err
				}
				mu.Unlock()
				return
			}

			resp.FrameId = frameID

			mu.Lock()
			sendErr := stream.Send(resp)
			mu.Unlock()

			if sendErr != nil {
				log.Printf("Frame %d send error: %v", frameID, sendErr)
			}
		}(frame)
	}

	wg.Wait()

	log.Printf("Video stream ended: %d frames processed", frameCount)

	if streamErr != nil {
		return status.Errorf(codes.Internal, "processing error: %v", streamErr)
	}

	return nil
}

// processFrame processes a single video frame.
func (s *Server) processFrame(
	ctx context.Context,
	frame *pb.VideoFrame,
) (*pb.PointCloudResponse, error) {
	startTime := time.Now()

	// Call inference service
	inferResp, err := s.inferenceClient.Infer(
		ctx,
		frame.ImageData,
		frame.Format,
		frame.Width,
		frame.Height,
		frame.Intrinsics,
	)
	if err != nil {
		return nil, err
	}

	// Convert to PLY points
	points, err := ply.PointsFromRaw(inferResp.PointCloudData, int(inferResp.NumPoints))
	if err != nil {
		return nil, err
	}

	originalCount := len(points)

	// Apply downsampling
	var downsampleTimeMs float64
	if frame.Downsampling != nil && frame.Downsampling.Enabled {
		voxelSize := frame.Downsampling.VoxelSize
		if voxelSize <= 0 {
			voxelSize = s.voxelConfig.VoxelSize
		}

		downsampler := voxel.NewWithSize(voxelSize)
		points, downsampleTimeMs = downsampler.Downsample(points)
	}

	// Write PLY
	writer := ply.NewBinaryWriter()
	plyData, err := writer.WriteToBytes(points)
	if err != nil {
		return nil, err
	}

	totalTimeMs := float32(time.Since(startTime).Milliseconds())

	return &pb.PointCloudResponse{
		PlyData:        plyData,
		NumPoints:      int32(len(points)),
		OriginalPoints: int32(originalCount),
		FrameId:        frame.FrameId,
		Stats: &pb.ProcessingStats{
			InferenceTimeMs:    inferResp.InferenceTimeMs,
			ProjectionTimeMs:   inferResp.ProjectionTimeMs,
			DownsamplingTimeMs: float32(downsampleTimeMs),
			TotalTimeMs:        totalTimeMs,
		},
	}, nil
}
