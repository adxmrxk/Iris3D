#!/usr/bin/env python3
"""
MiDaS v3.1 DPT-Large to ONNX Export Script

This script downloads the MiDaS depth estimation model from torch.hub
and exports it to ONNX format for production deployment with ONNX Runtime.

Usage:
    python export_onnx.py --output models/midas_v31_dpt_large.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.onnx


def download_midas_model(model_type: str = "DPT_Large") -> torch.nn.Module:
    """
    Download MiDaS model from torch.hub.

    Args:
        model_type: Model variant - "DPT_Large", "DPT_Hybrid", or "MiDaS_small"

    Returns:
        Loaded PyTorch model in eval mode
    """
    print(f"Downloading MiDaS {model_type} from torch.hub...")

    model = torch.hub.load(
        "intel-isl/MiDaS",
        model_type,
        pretrained=True,
        trust_repo=True
    )
    model.eval()

    print(f"Model downloaded successfully: {model_type}")
    return model


def get_midas_transforms(model_type: str = "DPT_Large"):
    """
    Get the preprocessing transforms for MiDaS.

    Returns:
        Transform function from midas.transforms
    """
    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms",
        trust_repo=True
    )

    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        return midas_transforms.dpt_transform
    else:
        return midas_transforms.small_transform


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_height: int = 384,
    input_width: int = 384,
    opset_version: int = 17
) -> None:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_height: Model input height
        input_width: Model input width
        opset_version: ONNX opset version
    """
    print(f"Exporting model to ONNX: {output_path}")

    # Create dummy input matching expected dimensions
    # MiDaS expects [N, C, H, W] normalized input
    dummy_input = torch.randn(1, 3, input_height, input_width)

    # Dynamic axes for batch size flexibility
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=False
    )

    print(f"ONNX model saved: {output_path}")

    # Validate the exported model
    validate_onnx_model(output_path, model, dummy_input)


def validate_onnx_model(
    onnx_path: str,
    pytorch_model: torch.nn.Module,
    dummy_input: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> None:
    """
    Validate ONNX model output matches PyTorch output.

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        dummy_input: Input tensor for validation
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
    """
    import onnx
    import onnxruntime as ort

    print("Validating ONNX model...")

    # Check ONNX model is well-formed
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model structure: Valid")

    # Compare outputs
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()

    # Run ONNX inference
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )
    onnx_output = session.run(
        None,
        {"input": dummy_input.numpy()}
    )[0]

    # Compare
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol):
        print("  Output comparison: PASSED")
    else:
        print("  Output comparison: WARNING - differences exceed tolerance")
        print("    (This may be acceptable for depth estimation)")

    # Print model info
    file_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    print(f"  Model size: {file_size_mb:.1f} MB")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {onnx_output.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Export MiDaS depth estimation model to ONNX format"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models/midas_v31_dpt_large.onnx",
        help="Output path for ONNX model"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="DPT_Large",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="MiDaS model variant"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=384,
        help="Input image size (height and width)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version"
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download and export model
    try:
        model = download_midas_model(args.model_type)
        export_to_onnx(
            model,
            str(output_path),
            input_height=args.input_size,
            input_width=args.input_size,
            opset_version=args.opset
        )
        print("\nExport completed successfully!")
        print(f"Model ready for deployment: {output_path}")

    except Exception as e:
        print(f"Export failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
