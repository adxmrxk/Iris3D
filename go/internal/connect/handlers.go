// Package connect provides Connect Protocol handlers for Iris3D.
// Connect Protocol enables HTTP/1.1, HTTP/2, and gRPC-compatible APIs.
package connect

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"
	"time"

	"connectrpc.com/connect"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/iris3d/go/internal/cache"
	"github.com/iris3d/go/internal/client"
	"github.com/iris3d/go/internal/ply"
	"github.com/iris3d/go/internal/telemetry"
	"github.com/iris3d/go/internal/voxel"

	iris3dv1 "github.com/iris3d/go/gen/iris3d/v1"
	"github.com/iris3d/go/gen/iris3d/v1/iris3dv1connect"
)

var tracer = otel.Tracer("iris3d-gateway")

// Iris3DHandler implements the Connect service interface.
type Iris3DHandler struct {
	iris3dv1connect.UnimplementedIris3DServiceHandler

	inferenceClient *client.InferenceClient
	cache           *cache.RedisCache
	metrics         *telemetry.Metrics
	defaultVoxel    float32
}

// Config for the handler.
type Config struct {
	InferenceAddress string
	RedisAddress     string
	DefaultVoxelSize float32
	CacheTTL         time.Duration
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		InferenceAddress: "localhost:50052",
		RedisAddress:     "localhost:6379",
		DefaultVoxelSize: 0.01,
		CacheTTL:         5 * time.Minute,
	}
}

// NewHandler creates a new Connect handler.
func NewHandler(cfg Config, metrics *telemetry.Metrics) (*Iris3DHandler, error) {
	// Create inference client
	clientCfg := client.DefaultConfig(cfg.InferenceAddress)
	inferenceClient, err := client.NewInferenceClient(clientCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create inference client: %w", err)
	}

	// Create Redis cache (optional)
	var redisCache *cache.RedisCache
	if cfg.RedisAddress != "" {
		redisCache, err = cache.NewRedisCache(cfg.RedisAddress, cfg.CacheTTL)
		if err != nil {
			log.Printf("Warning: Redis cache not available: %v", err)
		}
	}

	return &Iris3DHandler{
		inferenceClient: inferenceClient,
		cache:           redisCache,
		metrics:         metrics,
		defaultVoxel:    cfg.DefaultVoxelSize,
	}, nil
}

// Close cleans up resources.
func (h *Iris3DHandler) Close() error {
	if h.cache != nil {
		h.cache.Close()
	}
	return h.inferenceClient.Close()
}

// ProcessImage handles single image to point cloud conversion.
func (h *Iris3DHandler) ProcessImage(
	ctx context.Context,
	req *connect.Request[iris3dv1.ProcessImageRequest],
) (*connect.Response[iris3dv1.ProcessImageResponse], error) {
	// Start span
	ctx, span := tracer.Start(ctx, "ProcessImage",
		trace.WithAttributes(
			attribute.String("model", req.Msg.Model.String()),
			attribute.Int("image_size", len(req.Msg.ImageData)),
		),
	)
	defer span.End()

	startTime := time.Now()

	// Check cache
	cacheKey := h.computeCacheKey(req.Msg)
	if h.cache != nil {
		if cached, err := h.cache.Get(ctx, cacheKey); err == nil && cached != nil {
			span.SetAttributes(attribute.Bool("cache_hit", true))
			h.metrics.RecordCacheHit()
			return h.cachedResponse(cached)
		}
	}
	span.SetAttributes(attribute.Bool("cache_hit", false))

	// Call inference service
	inferResp, err := h.inferenceClient.Infer(
		ctx,
		req.Msg.ImageData,
		req.Msg.Format,
		req.Msg.Width,
		req.Msg.Height,
		h.convertIntrinsics(req.Msg.Intrinsics),
	)
	if err != nil {
		span.RecordError(err)
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	// Convert to PLY points
	points, err := ply.PointsFromRaw(inferResp.PointCloudData, int(inferResp.NumPoints))
	if err != nil {
		span.RecordError(err)
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	originalCount := len(points)

	// Apply processing options
	var downsampleTimeMs float64
	if opts := req.Msg.Options; opts != nil {
		// Voxel downsampling
		if opts.Downsampling != nil && opts.Downsampling.Enabled {
			voxelSize := opts.Downsampling.VoxelSize
			if voxelSize <= 0 {
				voxelSize = h.defaultVoxel
			}
			downsampler := voxel.NewWithSize(voxelSize)
			points, downsampleTimeMs = downsampler.Downsample(points)
		}
	}

	// Write PLY
	writer := ply.NewBinaryWriter()
	plyData, err := writer.WriteToBytes(points)
	if err != nil {
		span.RecordError(err)
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	totalTime := float32(time.Since(startTime).Milliseconds())

	// Record metrics
	h.metrics.RecordInference(req.Msg.Model.String(), float64(inferResp.InferenceTimeMs))
	h.metrics.RecordPointsGenerated(int64(len(points)))

	// Build response
	result := &iris3dv1.PointCloudResult{
		Data:              plyData,
		Format:            "ply_binary",
		NumPoints:         int32(len(points)),
		OriginalNumPoints: int32(originalCount),
		HasColors:         true,
		Stats: &iris3dv1.ProcessingStats{
			InferenceTimeMs:    inferResp.InferenceTimeMs,
			ProjectionTimeMs:   inferResp.ProjectionTimeMs,
			DownsamplingTimeMs: float32(downsampleTimeMs),
			TotalTimeMs:        totalTime,
			CacheHit:           false,
		},
	}

	// Cache result
	if h.cache != nil {
		h.cache.Set(ctx, cacheKey, plyData)
	}

	resp := &iris3dv1.ProcessImageResponse{
		Result: result,
	}

	log.Printf("ProcessImage: %d -> %d points in %.1fms",
		originalCount, len(points), totalTime)

	return connect.NewResponse(resp), nil
}

// GetModels returns available depth models.
func (h *Iris3DHandler) GetModels(
	ctx context.Context,
	req *connect.Request[iris3dv1.GetModelsRequest],
) (*connect.Response[iris3dv1.GetModelsResponse], error) {
	models := []*iris3dv1.ModelInfo{
		{
			Model:                iris3dv1.DepthModel_DEPTH_MODEL_MIDAS,
			Name:                 "MiDaS v3.1 DPT-Large",
			Description:          "Fast relative depth estimation with good generalization",
			IsMetric:             false,
			RecommendedInputSize: 384,
			EstimatedLatencyMs:   30,
		},
		{
			Model:                iris3dv1.DepthModel_DEPTH_MODEL_ZOEDEPTH,
			Name:                 "ZoeDepth",
			Description:          "Metric depth estimation with scale awareness",
			IsMetric:             true,
			RecommendedInputSize: 384,
			EstimatedLatencyMs:   50,
		},
		{
			Model:                iris3dv1.DepthModel_DEPTH_MODEL_DEPTH_ANYTHING_V2,
			Name:                 "Depth Anything v2 Large",
			Description:          "State-of-the-art depth estimation quality",
			IsMetric:             false,
			RecommendedInputSize: 518,
			EstimatedLatencyMs:   40,
		},
		{
			Model:                iris3dv1.DepthModel_DEPTH_MODEL_DEPTH_ANYTHING_V2_SMALL,
			Name:                 "Depth Anything v2 Small",
			Description:          "Fast variant with good quality balance",
			IsMetric:             false,
			RecommendedInputSize: 518,
			EstimatedLatencyMs:   20,
		},
	}

	return connect.NewResponse(&iris3dv1.GetModelsResponse{
		Models: models,
	}), nil
}

// Health returns service health status.
func (h *Iris3DHandler) Health(
	ctx context.Context,
	req *connect.Request[iris3dv1.HealthRequest],
) (*connect.Response[iris3dv1.HealthResponse], error) {
	details := make(map[string]string)
	details["gateway"] = "healthy"

	// Check inference service
	// TODO: Add actual health check

	// Check Redis
	if h.cache != nil {
		if err := h.cache.Ping(ctx); err != nil {
			details["cache"] = "unhealthy: " + err.Error()
		} else {
			details["cache"] = "healthy"
		}
	} else {
		details["cache"] = "disabled"
	}

	return connect.NewResponse(&iris3dv1.HealthResponse{
		Status:  iris3dv1.HealthResponse_STATUS_SERVING,
		Details: details,
	}), nil
}

// Helper methods

func (h *Iris3DHandler) computeCacheKey(req *iris3dv1.ProcessImageRequest) string {
	hash := sha256.New()
	hash.Write(req.ImageData)
	hash.Write([]byte(req.Model.String()))
	return "iris3d:pc:" + hex.EncodeToString(hash.Sum(nil))[:16]
}

func (h *Iris3DHandler) convertIntrinsics(in *iris3dv1.CameraIntrinsics) *client.CameraIntrinsics {
	if in == nil {
		return nil
	}
	return &client.CameraIntrinsics{
		FocalLengthX:    in.FocalLengthX,
		FocalLengthY:    in.FocalLengthY,
		PrincipalPointX: in.PrincipalPointX,
		PrincipalPointY: in.PrincipalPointY,
	}
}

func (h *Iris3DHandler) cachedResponse(data []byte) (*connect.Response[iris3dv1.ProcessImageResponse], error) {
	// Parse cached PLY to get point count
	// For simplicity, we'll estimate from data size
	numPoints := len(data) / 15 // Approximate: header + 15 bytes per point

	result := &iris3dv1.PointCloudResult{
		Data:      data,
		Format:    "ply_binary",
		NumPoints: int32(numPoints),
		HasColors: true,
		Stats: &iris3dv1.ProcessingStats{
			CacheHit: true,
		},
	}

	return connect.NewResponse(&iris3dv1.ProcessImageResponse{
		Result: result,
	}), nil
}
