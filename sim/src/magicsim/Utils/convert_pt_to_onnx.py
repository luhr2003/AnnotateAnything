#!/usr/bin/env python3
"""
Script to convert PyTorch TorchScript (.pt) model to ONNX format.

Usage:
    python scripts/convert_pt_to_onnx.py --input model.pt --output model.onnx --input_shape 1,516
"""

import argparse
import torch
import os


def convert_torchscript_to_onnx(
    pt_path: str,
    onnx_path: str,
    input_shape: tuple = (1, 516),
    opset_version: int = 18,
):
    """Convert TorchScript model to ONNX format.

    Args:
        pt_path: Path to the input .pt (TorchScript) model file
        onnx_path: Path to save the output .onnx file
        input_shape: Input shape for the model (batch_size, input_dim)
        opset_version: ONNX opset version to use (default: 18)
    """
    # Check if input file exists
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Input model file not found: {pt_path}")

    # Load TorchScript model
    print(f"Loading TorchScript model from: {pt_path}")
    model = torch.jit.load(pt_path, map_location="cpu")
    model.eval()

    # Create dummy input
    dummy_input = torch.zeros(input_shape, dtype=torch.float32)

    # Create output directory if needed
    os.makedirs(
        os.path.dirname(onnx_path) if os.path.dirname(onnx_path) else ".", exist_ok=True
    )

    # Export to ONNX
    print(f"Exporting to ONNX format: {onnx_path}")
    print(f"Input shape: {input_shape}, Opset version: {opset_version}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes=None,  # Fixed batch size
        verbose=False,
    )

    print(f"Successfully converted to ONNX: {onnx_path}")
    print(f"Model size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert TorchScript .pt model to ONNX format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input .pt (TorchScript) model file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .onnx file",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="1,516",
        help="Input shape as 'batch_size,input_dim' (default: '1,516')",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )

    args = parser.parse_args()

    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(",")))

    convert_torchscript_to_onnx(
        pt_path=args.input,
        onnx_path=args.output,
        input_shape=input_shape,
        opset_version=args.opset_version,
    )
