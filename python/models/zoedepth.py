"""
ZoeDepth Model

Metric depth estimation that outputs actual distances (meters).
https://github.com/isl-org/ZoeDepth
"""

from typing import Tuple, Optional, Literal
import numpy as np
import cv2

from .base import BaseDepthModel


class ZoeDepthModel(BaseDepthModel):
    """
    ZoeDepth model for metric depth estimation.

    Unlike MiDaS and Depth Anything, ZoeDepth outputs metric depth
    values (in meters), making it suitable for applications requiring
    actual distance measurements.

    Variants:
    - nyu: Trained on indoor scenes (NYU Depth v2)
    - kitti: Trained on outdoor driving scenes
    - nk: Combined training on both datasets
    """

    VARIANTS = {
        "nyu": "isl-org/ZoeDepth-NYU",
        "kitti": "isl-org/ZoeDepth-KITTI",
        "nk": "isl-org/ZoeDepth",
    }

    def __init__(
        self,
        variant: Literal["nyu", "kitti", "nk"] = "nk",
        model_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize ZoeDepth.

        Args:
            variant: Dataset variant (nyu, kitti, or nk for combined)
            model_path: Custom model path (optional)
            use_gpu: Use GPU for inference
        """
        super().__init__(use_gpu)
        self.variant = variant
        self.model_path = model_path
        self._model = None

    @property
    def name(self) -> str:
        return f"ZoeDepth ({self.variant.upper()})"

    @property
    def input_size(self) -> Tuple[int, int]:
        return (384, 512)  # ZoeDepth uses non-square input

    @property
    def is_metric(self) -> bool:
        return True

    def _load_model(self) -> None:
        """Load ZoeDepth model."""
        try:
            import torch

            hub_name = self.VARIANTS[self.variant]

            # Try HuggingFace first
            try:
                from transformers import AutoModelForDepthEstimation, AutoImageProcessor

                self._processor = AutoImageProcessor.from_pretrained(hub_name)
                self._model = AutoModelForDepthEstimation.from_pretrained(hub_name)
                self._use_hf = True

            except Exception:
                # Fallback to torch.hub
                self._model = torch.hub.load(
                    "isl-org/ZoeDepth",
                    "ZoeD_NK" if self.variant == "nk" else f"ZoeD_{self.variant.upper()}",
                    pretrained=True,
                    trust_repo=True
                )
                self._use_hf = False

            if self.use_gpu and torch.cuda.is_available():
                self._model = self._model.cuda()

            self._model.eval()

            device = "CUDA" if self.use_gpu and torch.cuda.is_available() else "CPU"
            print(f"ZoeDepth ({self.variant}) loaded on {device}")

        except ImportError:
            raise ImportError(
                "PyTorch required for ZoeDepth. Install with: "
                "pip install torch torchvision"
            )

    def _preprocess(self, image: np.ndarray):
        """Preprocess image for ZoeDepth."""
        import torch
        from PIL import Image

        # BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        if hasattr(self, '_use_hf') and self._use_hf:
            inputs = self._processor(images=pil_image, return_tensors="pt")
            return inputs
        else:
            # Direct tensor for torch.hub model
            return pil_image

    def _inference(self, input_data) -> np.ndarray:
        """Run ZoeDepth inference."""
        import torch

        with torch.no_grad():
            if hasattr(self, '_use_hf') and self._use_hf:
                if self.use_gpu and torch.cuda.is_available():
                    input_data = {k: v.cuda() for k, v in input_data.items()}

                outputs = self._model(**input_data)
                depth = outputs.predicted_depth
            else:
                # torch.hub model takes PIL image directly
                depth = self._model.infer_pil(input_data)

            return depth.cpu().numpy() if isinstance(depth, torch.Tensor) else depth

    def _postprocess(
        self,
        output: np.ndarray,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """Post-process metric depth output."""
        # Handle dimensions
        if output.ndim == 4:
            output = output[0, 0]
        elif output.ndim == 3:
            output = output[0]

        # ZoeDepth outputs metric depth, no normalization needed
        # Just resize to original dimensions
        depth = cv2.resize(
            output,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_CUBIC
        )

        return depth.astype(np.float32)
