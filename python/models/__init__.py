"""
Iris3D Model Registry

Provides a unified interface for multiple depth estimation models:
- MiDaS v3.1 (DPT-Large) - Fast, relative depth
- ZoeDepth - Metric depth estimation
- Depth Anything v2 - State-of-the-art quality
- SAM (Segment Anything) - Instance segmentation
"""

from typing import Dict, Type, Optional
from enum import Enum

from .base import BaseDepthModel, DepthResult
from .midas import MiDaSModel
from .depth_anything import DepthAnythingV2Model
from .zoedepth import ZoeDepthModel


class ModelType(Enum):
    """Available depth model types."""
    MIDAS = "midas"
    ZOEDEPTH = "zoedepth"
    DEPTH_ANYTHING_V2 = "depth_anything_v2"
    DEPTH_ANYTHING_V2_SMALL = "depth_anything_v2_small"


# Model registry
MODEL_REGISTRY: Dict[ModelType, Type[BaseDepthModel]] = {
    ModelType.MIDAS: MiDaSModel,
    ModelType.ZOEDEPTH: ZoeDepthModel,
    ModelType.DEPTH_ANYTHING_V2: DepthAnythingV2Model,
    ModelType.DEPTH_ANYTHING_V2_SMALL: DepthAnythingV2Model,
}


# Model metadata
MODEL_INFO = {
    ModelType.MIDAS: {
        "name": "MiDaS v3.1 DPT-Large",
        "description": "Fast relative depth estimation, good generalization",
        "is_metric": False,
        "input_size": 384,
        "latency_gpu_ms": 30,
        "latency_cpu_ms": 200,
    },
    ModelType.ZOEDEPTH: {
        "name": "ZoeDepth",
        "description": "Metric depth estimation with scale awareness",
        "is_metric": True,
        "input_size": 384,
        "latency_gpu_ms": 50,
        "latency_cpu_ms": 400,
    },
    ModelType.DEPTH_ANYTHING_V2: {
        "name": "Depth Anything v2 Large",
        "description": "State-of-the-art depth estimation quality",
        "is_metric": False,
        "input_size": 518,
        "latency_gpu_ms": 40,
        "latency_cpu_ms": 350,
    },
    ModelType.DEPTH_ANYTHING_V2_SMALL: {
        "name": "Depth Anything v2 Small",
        "description": "Fast variant with good quality",
        "is_metric": False,
        "input_size": 518,
        "latency_gpu_ms": 20,
        "latency_cpu_ms": 150,
    },
}


class ModelManager:
    """
    Manages multiple depth estimation models with lazy loading.

    Example:
        manager = ModelManager(use_gpu=True)
        model = manager.get_model(ModelType.DEPTH_ANYTHING_V2)
        depth_map, timing = model.estimate(image)
    """

    def __init__(self, use_gpu: bool = True, cache_models: bool = True):
        """
        Initialize the model manager.

        Args:
            use_gpu: Use GPU for inference if available
            cache_models: Keep loaded models in memory
        """
        self.use_gpu = use_gpu
        self.cache_models = cache_models
        self._models: Dict[ModelType, BaseDepthModel] = {}

    def get_model(self, model_type: ModelType) -> BaseDepthModel:
        """
        Get a depth model instance, loading if necessary.

        Args:
            model_type: Type of model to load

        Returns:
            Initialized depth model
        """
        if model_type in self._models:
            return self._models[model_type]

        model_class = MODEL_REGISTRY.get(model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {model_type}")

        # Handle small variant
        if model_type == ModelType.DEPTH_ANYTHING_V2_SMALL:
            model = model_class(variant="small", use_gpu=self.use_gpu)
        else:
            model = model_class(use_gpu=self.use_gpu)

        if self.cache_models:
            self._models[model_type] = model

        return model

    def unload_model(self, model_type: ModelType) -> None:
        """Unload a cached model to free memory."""
        if model_type in self._models:
            del self._models[model_type]

    def unload_all(self) -> None:
        """Unload all cached models."""
        self._models.clear()

    @staticmethod
    def get_model_info(model_type: ModelType) -> dict:
        """Get metadata for a model type."""
        return MODEL_INFO.get(model_type, {})

    @staticmethod
    def list_models() -> list:
        """List all available model types with their info."""
        return [
            {"type": mt.value, **MODEL_INFO[mt]}
            for mt in ModelType
        ]


__all__ = [
    "ModelType",
    "ModelManager",
    "BaseDepthModel",
    "DepthResult",
    "MiDaSModel",
    "ZoeDepthModel",
    "DepthAnythingV2Model",
]
