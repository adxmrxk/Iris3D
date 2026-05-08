"""
Segment Anything Model (SAM) Integration

Provides instance segmentation for selective 3D reconstruction.
https://github.com/facebookresearch/segment-anything
"""

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class SegmentationMask:
    """Single segmentation mask result."""

    # Binary mask (H, W) uint8, values 0 or 255
    mask: np.ndarray

    # Bounding box [x_min, y_min, x_max, y_max]
    bbox: Tuple[int, int, int, int]

    # Predicted IoU score
    predicted_iou: float

    # Stability score
    stability_score: float

    # Area in pixels
    area: int

    # Point that prompted this mask (if any)
    point_prompt: Optional[Tuple[int, int]] = None


@dataclass
class SegmentationResult:
    """Result from SAM segmentation."""

    masks: List[SegmentationMask]
    encoding_time_ms: float
    mask_generation_time_ms: float
    total_time_ms: float


class SAMModel:
    """
    Segment Anything Model wrapper.

    Supports:
    - Automatic mask generation (no prompts)
    - Point prompts (click to segment)
    - Box prompts (region of interest)

    Example:
        sam = SAMModel()

        # Automatic segmentation
        result = sam.segment(image)

        # Point prompt
        result = sam.segment(image, point_prompts=[(100, 200)])

        # Box prompt
        result = sam.segment(image, box_prompt=(50, 50, 200, 200))
    """

    VARIANTS = {
        "vit_h": "facebook/sam-vit-huge",
        "vit_l": "facebook/sam-vit-large",
        "vit_b": "facebook/sam-vit-base",
    }

    def __init__(
        self,
        variant: str = "vit_l",
        use_gpu: bool = True
    ):
        """
        Initialize SAM.

        Args:
            variant: Model variant (vit_h, vit_l, vit_b)
            use_gpu: Use GPU for inference
        """
        self.variant = variant
        self.use_gpu = use_gpu
        self._model = None
        self._processor = None
        self._is_initialized = False

    def initialize(self) -> None:
        """Load SAM model."""
        if self._is_initialized:
            return

        try:
            from transformers import SamModel, SamProcessor
            import torch

            hub_name = self.VARIANTS.get(self.variant, self.VARIANTS["vit_l"])

            self._processor = SamProcessor.from_pretrained(hub_name)
            self._model = SamModel.from_pretrained(hub_name)

            if self.use_gpu and torch.cuda.is_available():
                self._model = self._model.cuda()

            self._model.eval()
            self._is_initialized = True

            device = "CUDA" if self.use_gpu and torch.cuda.is_available() else "CPU"
            print(f"SAM ({self.variant}) loaded on {device}")

        except ImportError:
            raise ImportError(
                "transformers package required. Install with: "
                "pip install transformers torch"
            )

    def segment(
        self,
        image: np.ndarray,
        point_prompts: Optional[List[Tuple[int, int]]] = None,
        point_labels: Optional[List[int]] = None,
        box_prompt: Optional[Tuple[int, int, int, int]] = None,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_area: int = 100,
    ) -> SegmentationResult:
        """
        Segment image with SAM.

        Args:
            image: BGR image (H, W, 3) uint8
            point_prompts: List of (x, y) click points
            point_labels: Labels for points (1=foreground, 0=background)
            box_prompt: Bounding box (x_min, y_min, x_max, y_max)
            pred_iou_thresh: Minimum predicted IoU score
            stability_score_thresh: Minimum stability score
            min_mask_area: Minimum mask area in pixels

        Returns:
            SegmentationResult with masks
        """
        import time
        import torch
        from PIL import Image

        self.initialize()

        start_total = time.perf_counter()

        # Convert to RGB PIL image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Prepare inputs
        if point_prompts is not None:
            input_points = [[[p[0], p[1]] for p in point_prompts]]
            input_labels = [[l for l in (point_labels or [1] * len(point_prompts))]]
            inputs = self._processor(
                pil_image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt"
            )
        elif box_prompt is not None:
            input_boxes = [[[box_prompt[0], box_prompt[1], box_prompt[2], box_prompt[3]]]]
            inputs = self._processor(
                pil_image,
                input_boxes=input_boxes,
                return_tensors="pt"
            )
        else:
            # Automatic mask generation - use grid of points
            inputs = self._processor(pil_image, return_tensors="pt")

        encoding_start = time.perf_counter()

        with torch.no_grad():
            if self.use_gpu and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Get image embeddings
            image_embeddings = self._model.get_image_embeddings(inputs["pixel_values"])

        encoding_time = (time.perf_counter() - encoding_start) * 1000

        mask_start = time.perf_counter()

        with torch.no_grad():
            # Generate masks
            outputs = self._model(
                image_embeddings=image_embeddings,
                **{k: v for k, v in inputs.items() if k != "pixel_values"}
            )

            masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]

            scores = outputs.iou_scores.cpu().numpy()[0]

        mask_time = (time.perf_counter() - mask_start) * 1000
        total_time = (time.perf_counter() - start_total) * 1000

        # Process masks
        result_masks = []
        masks_np = masks.numpy()

        for i in range(masks_np.shape[0]):
            for j in range(masks_np.shape[1]):
                mask = (masks_np[i, j] > 0.5).astype(np.uint8) * 255
                score = float(scores[i, j])

                # Filter by thresholds
                if score < pred_iou_thresh:
                    continue

                area = int(mask.sum() / 255)
                if area < min_mask_area:
                    continue

                # Get bounding box
                coords = np.where(mask > 0)
                if len(coords[0]) == 0:
                    continue

                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()

                result_masks.append(SegmentationMask(
                    mask=mask,
                    bbox=(int(x_min), int(y_min), int(x_max), int(y_max)),
                    predicted_iou=score,
                    stability_score=score,  # Using IoU as proxy
                    area=area,
                    point_prompt=point_prompts[i] if point_prompts and i < len(point_prompts) else None
                ))

        return SegmentationResult(
            masks=result_masks,
            encoding_time_ms=encoding_time,
            mask_generation_time_ms=mask_time,
            total_time_ms=total_time
        )

    def apply_mask_to_depth(
        self,
        depth_map: np.ndarray,
        mask: np.ndarray,
        invert: bool = False
    ) -> np.ndarray:
        """
        Apply segmentation mask to depth map.

        Args:
            depth_map: Depth map (H, W) float32
            mask: Binary mask (H, W) uint8
            invert: If True, keep background instead of foreground

        Returns:
            Masked depth map
        """
        mask_bool = mask > 127
        if invert:
            mask_bool = ~mask_bool

        result = depth_map.copy()
        result[~mask_bool] = 0  # Set masked areas to zero depth

        return result
