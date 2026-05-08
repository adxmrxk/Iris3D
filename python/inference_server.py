#!/usr/bin/env python3
"""
Iris3D Python Inference Server

gRPC service that handles depth estimation and 3D projection.
This is the internal service called by the Go gateway.

Usage:
    python inference_server.py --config config.yaml
"""

import argparse
import sys
import time
from concurrent import futures
from pathlib import Path
from typing import Optional

import grpc
import yaml
import numpy as np

# Import generated protobuf modules
import iris3d_pb2
import iris3d_pb2_grpc

from depth_estimator import (
    DepthEstimator,
    decode_image,
    decode_raw_image
)
from projection import (
    CameraIntrinsics,
    project_to_3d,
    point_cloud_to_bytes
)


class InferenceServicer(iris3d_pb2_grpc.InferenceServiceServicer):
    """
    gRPC servicer for depth estimation and 3D projection.
    """

    def __init__(self, config: dict):
        """
        Initialize the inference servicer.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Load depth estimation model
        model_config = config.get("model", {})
        model_path = model_config.get("path", "models/midas_v31_dpt_large.onnx")

        print(f"Loading model from: {model_path}")
        self.estimator = DepthEstimator(
            model_path=model_path,
            input_height=model_config.get("input_height", 384),
            input_width=model_config.get("input_width", 384),
            use_gpu=model_config.get("use_gpu", True)
        )

        # Load default camera settings
        self.camera_config = config.get("camera", {}).get("defaults", {})
        self.depth_config = config.get("depth", {})

        print("InferenceServicer initialized successfully")

    def _get_intrinsics(
        self,
        request_intrinsics: Optional[iris3d_pb2.CameraIntrinsics],
        image_width: int,
        image_height: int
    ) -> CameraIntrinsics:
        """
        Get camera intrinsics, using request values or defaults.

        Args:
            request_intrinsics: Optional intrinsics from request
            image_width: Image width for default computation
            image_height: Image height for default computation

        Returns:
            CameraIntrinsics object
        """
        # Check if request provides intrinsics
        if request_intrinsics is not None:
            fx = request_intrinsics.focal_length_x
            fy = request_intrinsics.focal_length_y
            cx = request_intrinsics.principal_point_x
            cy = request_intrinsics.principal_point_y

            # If any value is provided and non-zero, use request intrinsics
            if fx > 0 or fy > 0 or cx > 0 or cy > 0:
                return CameraIntrinsics(
                    fx=fx if fx > 0 else float(image_width),
                    fy=fy if fy > 0 else float(image_width),
                    cx=cx if cx > 0 else image_width / 2.0,
                    cy=cy if cy > 0 else image_height / 2.0
                )

        # Use config defaults
        cfg_fx = self.camera_config.get("focal_length_x", 0)
        cfg_fy = self.camera_config.get("focal_length_y", 0)
        cfg_cx = self.camera_config.get("principal_point_x", 0)
        cfg_cy = self.camera_config.get("principal_point_y", 0)

        # If config values are 0, compute from image size
        return CameraIntrinsics(
            fx=cfg_fx if cfg_fx > 0 else float(image_width),
            fy=cfg_fy if cfg_fy > 0 else float(image_width),
            cx=cfg_cx if cfg_cx > 0 else image_width / 2.0,
            cy=cfg_cy if cfg_cy > 0 else image_height / 2.0
        )

    def Infer(
        self,
        request: iris3d_pb2.InferenceRequest,
        context: grpc.ServicerContext
    ) -> iris3d_pb2.InferenceResponse:
        """
        Process a single image and return 3D point cloud.

        Args:
            request: InferenceRequest with image data
            context: gRPC context

        Returns:
            InferenceResponse with point cloud data
        """
        total_start = time.perf_counter()

        try:
            # Decode image
            if request.format.lower() == "raw":
                image = decode_raw_image(
                    request.image_data,
                    request.width,
                    request.height
                )
            else:
                image = decode_image(request.image_data, request.format)

            image_height, image_width = image.shape[:2]

            # Get camera intrinsics
            intrinsics = self._get_intrinsics(
                request.intrinsics if request.HasField("intrinsics") else None,
                image_width,
                image_height
            )

            # Run depth estimation
            depth_config = self.depth_config
            depth_map, inference_time_ms = self.estimator.estimate(
                image,
                scale_factor=depth_config.get("scale_factor", 1.0),
                min_depth=depth_config.get("min_depth", 0.1),
                max_depth=depth_config.get("max_depth", 100.0)
            )

            # Project to 3D
            point_cloud, projection_time_ms = project_to_3d(
                depth_map,
                image,
                intrinsics,
                min_depth=depth_config.get("min_depth", 0.1),
                max_depth=depth_config.get("max_depth", 100.0)
            )

            # Serialize point cloud
            point_cloud_bytes = point_cloud_to_bytes(point_cloud)
            num_points = point_cloud.shape[0]

            total_time_ms = (time.perf_counter() - total_start) * 1000
            print(
                f"Inference complete: {num_points} points, "
                f"inference={inference_time_ms:.1f}ms, "
                f"projection={projection_time_ms:.1f}ms, "
                f"total={total_time_ms:.1f}ms"
            )

            return iris3d_pb2.InferenceResponse(
                point_cloud_data=point_cloud_bytes,
                num_points=num_points,
                image_width=image_width,
                image_height=image_height,
                inference_time_ms=inference_time_ms,
                projection_time_ms=projection_time_ms
            )

        except Exception as e:
            print(f"Inference error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return iris3d_pb2.InferenceResponse()


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def serve(config: dict):
    """
    Start the gRPC server.

    Args:
        config: Configuration dictionary
    """
    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 50052)
    max_workers = server_config.get("max_workers", 4)

    # Create server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]
    )

    # Add servicer
    servicer = InferenceServicer(config)
    iris3d_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)

    # Start server
    address = f"{host}:{port}"
    server.add_insecure_port(address)
    server.start()

    print(f"Inference server listening on {address}")
    print("Press Ctrl+C to stop")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop(grace=5)


def main():
    parser = argparse.ArgumentParser(
        description="Iris3D Python Inference Server"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(str(config_path))

    # Start server
    serve(config)


if __name__ == "__main__":
    main()
