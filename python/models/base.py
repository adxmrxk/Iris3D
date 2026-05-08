"""
Base classes for depth estimation models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass
class DepthResult:
    """Result from depth estimation."""

    # Depth map (H, W) float32
    depth_map: np.ndarray

    # Original image dimensions
    original_height: int
    original_width: int

    # Whether depth values are metric (meters)
    is_metric: bool

    # Depth value range
    min_depth: float
    max_depth: float

    # Processing time in milliseconds
    inference_time_ms: float

    # Model used
    model_name: str


class BaseDepthModel(ABC):
    """
    Abstract base class for depth estimation models.

    All depth models must implement the estimate() method.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the depth model.

        Args:
            use_gpu: Whether to use GPU for inference
        """
        self.use_gpu = use_gpu
        self._is_initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging/metrics."""
        pass

    @property
    @abstractmethod
    def input_size(self) -> Tuple[int, int]:
        """Expected input size (height, width)."""
        pass

    @property
    def is_metric(self) -> bool:
        """Whether this model outputs metric depth."""
        return False

    @abstractmethod
    def _load_model(self) -> None:
        """Load model weights and initialize inference engine."""
        pass

    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: BGR image (H, W, 3) uint8

        Returns:
            Preprocessed tensor ready for inference
        """
        pass

    @abstractmethod
    def _inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run model inference.

        Args:
            input_tensor: Preprocessed input

        Returns:
            Raw model output
        """
        pass

    @abstractmethod
    def _postprocess(
        self,
        output: np.ndarray,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Post-process model output to depth map.

        Args:
            output: Raw model output
            original_size: Original image (height, width)

        Returns:
            Depth map at original resolution
        """
        pass

    def initialize(self) -> None:
        """Initialize the model (lazy loading)."""
        if not self._is_initialized:
            self._load_model()
            self._is_initialized = True

    def estimate(
        self,
        image: np.ndarray,
        min_depth: float = 0.1,
        max_depth: float = 100.0
    ) -> DepthResult:
        """
        Estimate depth from an image.

        Args:
            image: BGR image (H, W, 3) uint8
            min_depth: Minimum depth threshold
            max_depth: Maximum depth threshold

        Returns:
            DepthResult with depth map and metadata
        """
        import time

        # Lazy initialization
        self.initialize()

        original_size = (image.shape[0], image.shape[1])

        # Time the inference
        start = time.perf_counter()

        # Preprocess
        input_tensor = self._preprocess(image)

        # Inference
        output = self._inference(input_tensor)

        # Postprocess
        depth_map = self._postprocess(output, original_size)

        inference_time_ms = (time.perf_counter() - start) * 1000

        # Clamp to valid range
        depth_map = np.clip(depth_map, min_depth, max_depth)

        return DepthResult(
            depth_map=depth_map,
            original_height=original_size[0],
            original_width=original_size[1],
            is_metric=self.is_metric,
            min_depth=float(depth_map.min()),
            max_depth=float(depth_map.max()),
            inference_time_ms=inference_time_ms,
            model_name=self.name,
        )
