#!/usr/bin/env python3
"""
Iris3D Client Example

Demonstrates how to use the Iris3D gRPC service to convert
2D images to 3D point clouds.

Usage:
    python client_example.py --image path/to/image.jpg --output output.ply
"""

import argparse
import sys
from pathlib import Path

import grpc

# Import generated protobuf modules
import iris3d_pb2
import iris3d_pb2_grpc


def process_image(
    image_path: str,
    output_path: str,
    server_address: str = "localhost:50051",
    enable_downsampling: bool = True,
    voxel_size: float = 0.01
) -> None:
    """
    Process a single image and save the resulting point cloud.

    Args:
        image_path: Path to input image
        output_path: Path to save PLY file
        server_address: gRPC server address
        enable_downsampling: Whether to apply voxel downsampling
        voxel_size: Voxel size for downsampling
    """
    # Read image file
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as f:
        image_data = f.read()

    # Determine format from extension
    ext = image_path.suffix.lower()
    format_map = {
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".png": "png",
    }
    image_format = format_map.get(ext, "jpeg")

    # Create gRPC channel
    channel = grpc.insecure_channel(
        server_address,
        options=[
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ]
    )

    # Create stub
    stub = iris3d_pb2_grpc.Iris3DStub(channel)

    # Create request
    request = iris3d_pb2.ImageRequest(
        image_data=image_data,
        format=image_format,
        downsampling=iris3d_pb2.DownsamplingConfig(
            enabled=enable_downsampling,
            voxel_size=voxel_size
        )
    )

    print(f"Processing image: {image_path}")
    print(f"  Format: {image_format}")
    print(f"  Size: {len(image_data)} bytes")
    print(f"  Downsampling: {'enabled' if enable_downsampling else 'disabled'}")

    # Call service
    try:
        response = stub.ProcessImage(request)
    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()} - {e.details()}")
        sys.exit(1)

    # Save PLY file
    output_path = Path(output_path)
    with open(output_path, "wb") as f:
        f.write(response.ply_data)

    # Print statistics
    print(f"\nResults:")
    print(f"  Output: {output_path}")
    print(f"  Original points: {response.original_points:,}")
    print(f"  Final points: {response.num_points:,}")
    print(f"  PLY size: {len(response.ply_data):,} bytes")

    if response.stats:
        print(f"\nTiming:")
        print(f"  Inference: {response.stats.inference_time_ms:.1f} ms")
        print(f"  Projection: {response.stats.projection_time_ms:.1f} ms")
        print(f"  Downsampling: {response.stats.downsampling_time_ms:.1f} ms")
        print(f"  Total: {response.stats.total_time_ms:.1f} ms")

    print(f"\nPoint cloud saved to: {output_path}")
    print("Open with MeshLab, CloudCompare, or Open3D to visualize.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 2D image to 3D point cloud using Iris3D"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input image (JPEG or PNG)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.ply",
        help="Output PLY file path"
    )
    parser.add_argument(
        "--server", "-s",
        type=str,
        default="localhost:50051",
        help="Iris3D server address"
    )
    parser.add_argument(
        "--no-downsample",
        action="store_true",
        help="Disable voxel downsampling"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Voxel size for downsampling"
    )

    args = parser.parse_args()

    process_image(
        image_path=args.image,
        output_path=args.output,
        server_address=args.server,
        enable_downsampling=not args.no_downsample,
        voxel_size=args.voxel_size
    )


if __name__ == "__main__":
    main()
