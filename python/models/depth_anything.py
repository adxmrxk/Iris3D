"""
Depth Anything v2 Model

State-of-the-art monocular depth estimation with excellent detail preservation.
https://github.com/DepthAnything/Depth-Anything-V2
"""

from typing import Tuple, Optional, Literal
import numpy as np
import cv2

from .base import BaseDepthModel


class DepthAnythingV2Model(BaseDepthModel):
    """
    Depth Anything v2 model for high-quality depth estimation.

    Supports multiple model sizes:
    - small: Fastest, good for real-time
    - base: Balanced speed/quality
    - large: Best quality (default)
    """

    VARIANTS = {
        "small": {
            "input_size": 518,
            "hub_name": "depth-anything/Depth-Anything-V2-Small-hf",
        },
        "base": {
            "input_size": 518,
            "hub_name": "depth-anything/Depth-Anything-V2-Base-hf",
        },
        "large": {
            "input_size": 518,
            "hub_name": "depth-anything/Depth-Anything-V2-Large-hf",
        },
    }

    def __init__(
        self,
        variant: Literal["small", "base", "large"] = "large",
        model_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize Depth Anything v2.

        Args:
            variant: Model size variant
            model_path: Custom ONNX model path (optional)
            use_gpu: Use GPU for inference
        """
        super().__init__(use_gpu)
        self.variant = variant
        self.config = self.VARIANTS[variant]
        self.model_path = model_path
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return f"Depth Anything v2 {self.variant.capitalize()}"

    @property
    def input_size(self) -> Tuple[int, int]:
        size = self.config["input_size"]
        return (size, size)

    def _load_model(self) -> None:
        """Load model from HuggingFace or ONNX."""
        if self.model_path:
            self._load_onnx_model()
        else:
            self._load_hf_model()

    def _load_hf_model(self) -> None:
        """Load from HuggingFace transformers."""
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            import torch

            hub_name = self.config["hub_name"]

            self._processor = AutoImageProcessor.from_pretrained(hub_name)
            self._model = AutoModelForDepthEstimation.from_pretrained(hub_name)

            if self.use_gpu and torch.cuda.is_available():
                self._model = self._model.cuda()

            self._model.eval()
            self._use_transformers = True

            device = "CUDA" if self.use_gpu and torch.cuda.is_available() else "CPU"
            print(f"Depth Anything v2 ({self.variant}) loaded on {device}")

        except ImportError:
            raise ImportError(
                "transformers package required. Install with: "
                "pip install transformers torch"
            )

    def _load_onnx_model(self) -> None:
        """Load from ONNX file."""
        import onnxruntime as ort

        providers = []
        if self.use_gpu:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(
            self.model_path,
            providers=providers
        )
        self._use_transformers = False
        print(f"Depth Anything v2 ONNX loaded: {self.model_path}")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image."""
        if hasattr(self, '_use_transformers') and self._use_transformers:
            # HuggingFace preprocessing
            from PIL import Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            inputs = self._processor(images=pil_image, return_tensors="pt")
            return inputs
        else:
            # Manual preprocessing for ONNX
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(
                image_rgb,
                self.input_size[::-1],  # (width, height)
                interpolation=cv2.INTER_CUBIC
            )

            # Normalize
            image_float = image_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_normalized = (image_float - mean) / std

            # NCHW
            image_tensor = np.transpose(image_normalized, (2, 0, 1))
            image_tensor = np.expand_dims(image_tensor, axis=0)

            return image_tensor.astype(np.float32)

    def _inference(self, input_tensor) -> np.ndarray:
        """Run inference."""
        if hasattr(self, '_use_transformers') and self._use_transformers:
            import torch

            with torch.no_grad():
                if self.use_gpu and torch.cuda.is_available():
                    input_tensor = {k: v.cuda() for k, v in input_tensor.items()}

                outputs = self._model(**input_tensor)
                depth = outputs.predicted_depth

                return depth.cpu().numpy()
        else:
            # ONNX inference
            input_name = self._session.get_inputs()[0].name
            output_name = self._session.get_outputs()[0].name
            outputs = self._session.run([output_name], {input_name: input_tensor})
            return outputs[0]

    def _postprocess(
        self,
        output: np.ndarray,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """Post-process depth output."""
        # Handle batch dimension
        if output.ndim == 4:
            output = output[0, 0]
        elif output.ndim == 3:
            output = output[0]

        # Normalize to [0, 1]
        depth_min, depth_max = output.min(), output.max()
        if depth_max - depth_min > 1e-6:
            depth = (output - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(output)

        # Resize to original
        depth = cv2.resize(
            depth,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_CUBIC
        )

        return depth.astype(np.float32)
