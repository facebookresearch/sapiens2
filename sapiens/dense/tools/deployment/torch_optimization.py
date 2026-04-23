# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch model optimization utilities for exporting and compiling models.

This module provides utilities for:
- Converting SyncBatchNorm layers to standard BatchNorm
- Benchmarking model performance
- Exporting models with torch.export
- Compiling models with torch.compile
"""

import argparse
from typing import Any

import numpy as np
import torch
from sapiens.dense.models import init_model
from torch import nn


# =============================================================================
# BatchNorm Conversion Utilities
# =============================================================================


class _BatchNormXd(nn.modules.batchnorm._BatchNorm):
    """A general BatchNorm layer without input dimension check.

    Reproduced from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)

    The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
    is `_check_input_dim` that is designed for tensor sanity checks.
    The check has been bypassed in this class for the convenience of converting
    SyncBatchNorm.
    """

    def _check_input_dim(self, input: torch.Tensor) -> None:
        return


def revert_sync_batchnorm(module: nn.Module) -> nn.Module:
    """Convert all SyncBatchNorm layers in the model to BatchNormXd layers.

    Adapted from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)

    Args:
        module: The module containing `SyncBatchNorm` layers.

    Returns:
        The converted module with `BatchNormXd` layers.
    """
    module_output = module
    module_checklist = [torch.nn.modules.batchnorm.SyncBatchNorm]

    if isinstance(module, tuple(module_checklist)):
        module_output = _BatchNormXd(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        module_output.training = module.training
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        try:
            module_output.add_module(name, revert_sync_batchnorm(child))
        except Exception:
            print(f"Failed to convert {child} from SyncBN to BN!")

    del module
    return module_output


def convert_batchnorm(module: nn.Module) -> nn.Module:
    """Convert SyncBatchNorm to BatchNorm2d and optionally SiLU to ReLU.

    Args:
        module: The module to convert.

    Returns:
        The converted module.
    """
    module_output = module

    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked

    if isinstance(module, torch.nn.SiLU):
        module_output = torch.nn.ReLU(inplace=True)

    for name, child in module.named_children():
        module_output.add_module(name, convert_batchnorm(child))

    del module
    return module_output


# =============================================================================
# Benchmarking Utilities
# =============================================================================


def benchmark_model(
    model: nn.Module,
    inputs: dict[str, Any],
    model_name: str = "",
    num_warmup: int = 3,
    num_iterations: int = 10,
) -> float:
    """Benchmark model inference time.

    Args:
        model: The model to benchmark.
        inputs: Dictionary containing 'imgs' tensor.
        model_name: Name for logging purposes.
        num_warmup: Number of warmup iterations (not counted).
        num_iterations: Number of timed iterations.

    Returns:
        Mean inference time per sample in milliseconds.
    """
    imgs = (
        inputs["imgs"][0, ...].unsqueeze(0)
        if model_name.lower() == "original"
        else inputs["imgs"]
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream), torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            model(imgs)
            torch.cuda.synchronize()

        # Timed iterations
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_event.record()
            model(imgs)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))

    torch.cuda.current_stream().wait_stream(stream)

    mean_time = np.mean(times) / imgs.shape[0]
    print(f"Benchmark results for '{model_name}':")
    print(f"  Average time per sample: {mean_time:.2f} ms")
    print(f"  Total time ({num_iterations} iterations): {sum(times):.2f} ms")
    print(f"  Individual times: {[f'{t:.2f}' for t in times]}")

    return mean_time


# =============================================================================
# Input Generation
# =============================================================================


def create_demo_inputs(input_shape: tuple[int, int, int, int]) -> dict[str, Any]:
    """Create demo inputs for model testing and export.

    Args:
        input_shape: Tuple of (N, C, H, W) for input dimensions.

    Returns:
        Dictionary with 'imgs' tensor and 'img_metas' list.
    """
    n, c, h, w = input_shape
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)

    img_metas = [
        {
            "img_shape": (h, w, c),
            "ori_shape": (h, w, c),
            "pad_shape": (h, w, c),
            "filename": "<demo>.png",
            "scale_factor": 1.0,
            "flip": False,
        }
        for _ in range(n)
    ]

    return {
        "imgs": torch.FloatTensor(imgs),
        "img_metas": img_metas,
    }


# =============================================================================
# Model Export and Compilation
# =============================================================================


class _ToDeviceTransformer(torch.fx.Transformer):
    """FX Transformer to move operations to a specific device."""

    def __init__(self, module: nn.Module, device: str):
        super().__init__(module)
        self.target_device = torch.device(device)

    def call_function(self, target, args, kwargs):
        if "device" not in kwargs:
            return super().call_function(target, args, kwargs)

        kwargs = dict(kwargs)
        kwargs["device"] = self.target_device
        return super().call_function(target, args, kwargs)


def compile_and_export_model(
    model: nn.Module,
    inputs: dict[str, Any],
    output_file: str = "compiled_model.pt",
    max_batch_size: int = 32,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Export model using torch.export and optionally compile with torch.compile.

    Args:
        model: The model to export.
        inputs: Demo inputs for tracing.
        output_file: Path to save the exported model.
        max_batch_size: Maximum batch size for dynamic shapes.
        dtype: Data type for the model.
    """
    inputs["imgs"] = inputs["imgs"].to(dtype)
    imgs = inputs["imgs"]
    model.eval()

    # Define dynamic shapes
    dynamic_batch = torch.export.Dim("batch", min=1, max=max_batch_size)
    dynamic_h = torch.export.Dim("h", min=1024, max=2048)
    dynamic_w = torch.export.Dim("w", min=768, max=1536)
    dynamic_shapes = {"inputs": {0: dynamic_batch, 2: dynamic_h, 3: dynamic_w}}

    # Export model
    exported_model = torch.export.export(
        model,
        args=(imgs,),
        kwargs={},
        dynamic_shapes=dynamic_shapes,
    )
    torch.export.save(exported_model, output_file)
    print(f"Model exported to: {output_file}")

    if not torch.cuda.is_available():
        return
    # Compile and benchmark
    device = "cuda:0"
    model = torch.export.load(output_file).module().to(device)
    model = _ToDeviceTransformer(model, device).transform()
    imgs = imgs.to(device)
    inputs["imgs"] = inputs["imgs"].to(device)

    _compile_and_benchmark(model, imgs, inputs)


def _compile_and_benchmark(
    model: nn.Module,
    imgs: torch.Tensor,
    inputs: dict[str, Any],
) -> None:
    """Compile model and benchmark different compilation modes.

    Args:
        model: Model to compile.
        imgs: Input images tensor.
        inputs: Full inputs dictionary for benchmarking.
    """
    modes = {"default": "default"}
    best_mode = None
    min_mean = float("inf")

    for mode_name, mode in modes.items():
        print(f"Compiling model with '{mode_name}' mode...")

        compiled_model = torch.compile(model, mode=mode)

        # Warmup
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                compiled_model(imgs)
                torch.cuda.synchronize()
        torch.cuda.current_stream().wait_stream(stream)

        mean_time = benchmark_model(compiled_model, inputs, model_name=mode_name)
        if mean_time < min_mean:
            min_mean = mean_time
            best_mode = mode_name

    print(f"Best compilation mode: {best_mode}")


# =============================================================================
# CLI Interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Export and optimize a model for deployment"
    )
    parser.add_argument("config", help="Model config file path")
    parser.add_argument("--checkpoint", help="Checkpoint file path")
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1024, 768],
        help="Input image size as (height, width)",
    )
    parser.add_argument(
        "--output-file",
        "--output-dir",
        type=str,
        required=True,
        help="Output file path for exported model",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size for dynamic export",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 instead of bfloat16",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for model optimization CLI."""
    args = parse_args()

    # Determine input shape
    if len(args.shape) == 1:
        input_shape = (16, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (16, 3, args.shape[0], args.shape[1])
    else:
        raise ValueError("Shape must be 1 or 2 integers (height, width)")

    # Clamp batch size
    max_batch_size = args.max_batch_size
    input_shape = (max(1, min(input_shape[0], max_batch_size)), *input_shape[1:])

    # Initialize model
    model = init_model(args.config, args.checkpoint, device="cpu")
    model.eval()
    model = revert_sync_batchnorm(model)

    # Create demo inputs
    demo_inputs = create_demo_inputs(input_shape)

    # Set dtype
    dtype = torch.half if args.fp16 else torch.bfloat16
    model.to(dtype)
    demo_inputs["imgs"] = demo_inputs["imgs"].to(dtype)

    # Export and compile
    compile_and_export_model(
        model,
        demo_inputs,
        output_file=args.output_file,
        max_batch_size=max_batch_size,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
