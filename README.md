# Iris3D

A high-performance polyglot microservice that converts 2D RGB images into 3D point clouds (.PLY format) using deep learning depth estimation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Client                                     │
│              (Sends image/video frames via gRPC)                    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Go API Gateway (Port 50051)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐ │
│  │ gRPC Server │──│ Orchestrator │──│ Voxel Downsampling (Go)    │ │
│  │ (Streaming) │  │              │  │ + PLY Writer               │ │
│  └─────────────┘  └──────────────┘  └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼ gRPC (internal)
┌─────────────────────────────────────────────────────────────────────┐
│                  Python Inference Service (Port 50052)               │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────────────────┐ │
│  │ ONNX Runtime │──│ Depth Estimator│──│ Pinhole 3D Projection   │ │
│  │ (MiDaS v3.1) │  │                │  │ (Generates XYZ + RGB)   │ │
│  └──────────────┘  └────────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| API Gateway | Go + gRPC | High-performance request handling, streaming |
| Inference | Python + ONNX Runtime | ML model inference |
| Depth Model | MiDaS v3.1 DPT-Large | Monocular depth estimation |
| Downsampling | Go (Voxel Grid) | Point cloud optimization |
| Output Format | Binary PLY | Compact 3D representation |
| Orchestration | Docker Compose | Container management |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for model export)
- Go 1.21+ (for local development)
- NVIDIA GPU with CUDA (optional, for faster inference)

### 1. Export the ONNX Model

```bash
# Install Python dependencies
cd python
pip install -r requirements.txt

# Export MiDaS to ONNX (downloads ~400MB model)
python export_onnx.py --output ../models/midas_v31_dpt_large.onnx
```

### 2. Generate Protocol Buffers

```bash
make proto
```

### 3. Start Services with Docker

```bash
# Build and start containers
make docker

# Check logs
docker-compose logs -f
```

### 4. Test the Service

```bash
# Using the Python client
python python/client_example.py --image sample.jpg --output output.ply

# Using grpcurl
grpcurl -plaintext localhost:50051 list
```

## API Reference

### ProcessImage

Process a single image and return a 3D point cloud.

```protobuf
rpc ProcessImage(ImageRequest) returns (PointCloudResponse);
```

**Request:**
```json
{
  "image_data": "<base64 encoded image>",
  "format": "jpeg",
  "downsampling": {
    "enabled": true,
    "voxel_size": 0.01
  }
}
```

**Response:**
```json
{
  "ply_data": "<binary PLY>",
  "num_points": 150000,
  "original_points": 2073600,
  "stats": {
    "inference_time_ms": 45.2,
    "projection_time_ms": 12.1,
    "downsampling_time_ms": 8.5,
    "total_time_ms": 78.3
  }
}
```

### ProcessVideoStream

Process video frames as a bidirectional stream.

```protobuf
rpc ProcessVideoStream(stream VideoFrame) returns (stream PointCloudResponse);
```

## Configuration

### Python Service (`python/config.yaml`)

```yaml
server:
  host: "0.0.0.0"
  port: 50052
  max_workers: 4

model:
  path: "/app/models/midas_v31_dpt_large.onnx"
  input_height: 384
  input_width: 384
  use_gpu: true

camera:
  defaults:
    focal_length_x: 0  # Auto-compute from image
    focal_length_y: 0
    principal_point_x: 0
    principal_point_y: 0

depth:
  scale_factor: 1.0
  min_depth: 0.1
  max_depth: 100.0
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INFERENCE_HOST` | Python service address | `localhost:50052` |
| `PORT` | Gateway listen port | `50051` |

## Pinhole Camera Model

The 3D projection uses the standard pinhole camera model:

```
For each pixel (u, v) with depth Z:
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

Where:
    (fx, fy) = focal lengths in pixels
    (cx, cy) = principal point (image center)
```

By default, `fx = image_width` and `(cx, cy) = (width/2, height/2)`.

## Voxel Downsampling

The Go gateway performs voxel grid downsampling to reduce point cloud density:

1. Divide 3D space into a grid of voxel cells
2. For each voxel, keep only one representative point
3. Return the downsampled point cloud

This is essential for real-time applications and reduces file sizes by 10-100x.

## Development

### Local Development

```bash
# Terminal 1: Run Python inference service
make run-inference

# Terminal 2: Run Go gateway
make run-gateway
```

### Running Tests

```bash
make test
```

### Building

```bash
# Build Go binary
make build

# Build for Linux (cross-compile)
make build-go-linux
```

## Performance

| Metric | Value |
|--------|-------|
| Inference (GPU) | ~30-50ms |
| Inference (CPU) | ~200-500ms |
| 3D Projection | ~10-20ms |
| Voxel Downsampling | ~5-15ms |
| **Total (GPU)** | **~50-100ms** |

## Viewing Point Clouds

The output PLY files can be viewed with:

- [MeshLab](https://www.meshlab.net/) - Free, cross-platform
- [CloudCompare](https://www.danielgm.net/cc/) - Free, feature-rich
- [Open3D](http://www.open3d.org/) - Python library

## License

MIT
