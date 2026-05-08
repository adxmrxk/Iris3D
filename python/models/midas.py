"""
MiDaS v3.1 DPT-Large Depth Estimation Model

A well-established depth estimation model with good generalization.
Outputs relative (not metric) depth.
"""

from typing import Tuple, Optional
import numpy as np
import cv2

from .base import BaseDepthModel


class MiDaSModel(BaseDepthModel):
    """
    MiDaS v3.1 DPT-Large model for depth estimation.

    Uses ONNX Runtime for inference, optimized for production deployment.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = True,
        input_size: int = 384
    ):
        """
        Initialize MiDaS model.

        Args:
            model_path: Path to ONNX model file
            use_gpu: Use GPU for inference
            input_size: Model input resolution
        """
        super().__init__(use_gpu)
        self.model_path = model_path or "models/midas_v31_dpt_large.onnx"
        self._input_size = input_size
        self.session = None

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @property
    def name(self) -> str:
        return "MiDaS v3.1 DPT-Large"

    @property
    def input_size(self) -> Tuple[int, int]:
        return (self._input_size, self._input_size)

    def _load_model(self) -> None:
        """Load ONNX model."""
        import onnxruntime as ort

        providers = []
        if self.use_gpu:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            elif "DmlExecutionProvider" in available:
                providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"MiDaS loaded with provider: {self.session.get_providers()[0]}")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for MiDaS."""
        # BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image_resized = cv2.resize(
            image_rgb,
            (self._input_size, self._input_size),
            interpolation=cv2.INTER_CUBIC
        )

        # Normalize
        image_float = image_resized.astype(np.float32) / 255.0
        image_normalized = (image_float - self.mean) / self.std

        # NCHW format
        image_tensor = np.transpose(image_normalized, (2, 0, 1))
        image_tensor = np.expand_dims(image_tensor, axis=0)

        return image_tensor.astype(np.float32)

    def _inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        return outputs[0]

    def _postprocess(
        self,
        output: np.ndarray,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """Convert inverse depth to depth map."""
        # Remove batch dimension
        if output.ndim == 3:
            output = output[0]

        # MiDaS outputs inverse depth
        depth = 1.0 / (output + 1e-6)

        # Normalize to [0, 1] then scale
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max - depth_min > 1e-6:
            depth = (depth - depth_min) / (depth_max - depth_min)

        # Resize to original
        depth = cv2.resize(
            depth,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_CUBIC
        )

        return depth.astype(np.float32)
