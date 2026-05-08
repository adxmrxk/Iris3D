"""
Depth Estimator Module

Handles ONNX model loading and depth map inference using ONNX Runtime.
Optimized for production deployment with GPU support.
"""

import time
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import onnxruntime as ort


class DepthEstimator:
    """
    ONNX-based depth estimation using MiDaS v3.1.

    Handles image preprocessing, model inference, and depth map post-processing.
    """

    def __init__(
        self,
        model_path: str,
        input_height: int = 384,
        input_width: int = 384,
        use_gpu: bool = True
    ):
        """
        Initialize the depth estimator.

        Args:
            model_path: Path to ONNX model file
            input_height: Model input height
            input_width: Model input width
            use_gpu: Whether to use GPU for inference
        """
        self.model_path = Path(model_path)
        self.input_height = input_height
        self.input_width = input_width

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Configure ONNX Runtime session
        self.session = self._create_session(use_gpu)

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Normalization parameters (ImageNet stats used by MiDaS)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _create_session(self, use_gpu: bool) -> ort.InferenceSession:
        """
        Create ONNX Runtime inference session with appropriate providers.

        Args:
            use_gpu: Whether to attempt GPU acceleration

        Returns:
            Configured InferenceSession
        """
        providers = []

        if use_gpu:
            # Try CUDA first, then DirectML (Windows), then CPU
            available = ort.get_available_providers()

            if "CUDAExecutionProvider" in available:
                providers.append(("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                }))
            elif "DmlExecutionProvider" in available:
                providers.append("DmlExecutionProvider")

        providers.append("CPUExecutionProvider")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # Enable memory pattern optimization
        session_options.enable_mem_pattern = True

        session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=providers
        )

        active_provider = session.get_providers()[0]
        print(f"DepthEstimator initialized with provider: {active_provider}")

        return session

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for MiDaS inference.

        Args:
            image: Input BGR image (H, W, 3) uint8

        Returns:
            Tuple of (preprocessed tensor [1, 3, H, W], original size (H, W))
        """
        original_size = (image.shape[0], image.shape[1])

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        image_resized = cv2.resize(
            image_rgb,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_CUBIC
        )

        # Normalize to [0, 1]
        image_float = image_resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        image_normalized = (image_float - self.mean) / self.std

        # Transpose to NCHW format [1, 3, H, W]
        image_tensor = np.transpose(image_normalized, (2, 0, 1))
        image_tensor = np.expand_dims(image_tensor, axis=0)

        return image_tensor.astype(np.float32), original_size

    def postprocess(
        self,
        depth_output: np.ndarray,
        original_size: Tuple[int, int],
        scale_factor: float = 1.0,
        min_depth: float = 0.1,
        max_depth: float = 100.0
    ) -> np.ndarray:
        """
        Post-process depth map output.

        MiDaS outputs inverse relative depth. This method converts to
        metric-like depth and resizes to original image dimensions.

        Args:
            depth_output: Raw model output [1, H, W] or [H, W]
            original_size: Original image size (H, W)
            scale_factor: Depth scaling factor
            min_depth: Minimum depth threshold
            max_depth: Maximum depth threshold

        Returns:
            Depth map at original resolution (H, W) float32
        """
        # Remove batch dimension if present
        if depth_output.ndim == 3:
            depth_output = depth_output[0]

        # MiDaS outputs inverse depth - convert to depth
        # Add small epsilon to avoid division by zero
        depth_map = 1.0 / (depth_output + 1e-6)

        # Normalize to reasonable range
        depth_min = depth_map.min()
        depth_max = depth_map.max()

        if depth_max - depth_min > 1e-6:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map = np.zeros_like(depth_map)

        # Scale to desired depth range
        depth_map = depth_map * (max_depth - min_depth) + min_depth
        depth_map = depth_map * scale_factor

        # Resize to original image size
        depth_map = cv2.resize(
            depth_map,
            (original_size[1], original_size[0]),  # (width, height)
            interpolation=cv2.INTER_CUBIC
        )

        return depth_map.astype(np.float32)

    def estimate(
        self,
        image: np.ndarray,
        scale_factor: float = 1.0,
        min_depth: float = 0.1,
        max_depth: float = 100.0
    ) -> Tuple[np.ndarray, float]:
        """
        Run depth estimation on an image.

        Args:
            image: Input BGR image (H, W, 3) uint8
            scale_factor: Depth scaling factor
            min_depth: Minimum depth threshold
            max_depth: Maximum depth threshold

        Returns:
            Tuple of (depth map (H, W) float32, inference time in ms)
        """
        # Preprocess
        input_tensor, original_size = self.preprocess(image)

        # Run inference
        start_time = time.perf_counter()
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Postprocess
        depth_map = self.postprocess(
            outputs[0],
            original_size,
            scale_factor=scale_factor,
            min_depth=min_depth,
            max_depth=max_depth
        )

        return depth_map, inference_time_ms


def load_image(path: str) -> np.ndarray:
    """
    Load an image from file.

    Args:
        path: Path to image file

    Returns:
        BGR image as numpy array
    """
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


def decode_image(data: bytes, format: str = "jpeg") -> np.ndarray:
    """
    Decode image from bytes.

    Args:
        data: Encoded image bytes
        format: Image format ("jpeg", "png")

    Returns:
        BGR image as numpy array
    """
    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image data")
    return image


def decode_raw_image(
    data: bytes,
    width: int,
    height: int,
    channels: int = 3
) -> np.ndarray:
    """
    Decode raw RGB image from bytes.

    Args:
        data: Raw RGB bytes
        width: Image width
        height: Image height
        channels: Number of channels (default 3 for RGB)

    Returns:
        BGR image as numpy array
    """
    expected_size = width * height * channels
    if len(data) != expected_size:
        raise ValueError(
            f"Data size mismatch: expected {expected_size}, got {len(data)}"
        )

    # Reshape raw bytes to image
    image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, channels))

    # Convert RGB to BGR for OpenCV compatibility
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image
