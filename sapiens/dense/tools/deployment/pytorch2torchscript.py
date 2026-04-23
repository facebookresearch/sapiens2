# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# originally copied from https://www.internalfb.com/code/fbsource/[671aa4920700]/fbcode/xrcia/projects/sapiens/experimental_ghe_import/sapiens2/sapiens/seg/tools/deployment/pytorch2torchscript.py?lines=1-204

import argparse
import os

import torch
import torch._C
import torch.serialization
from sapiens.dense.models import init_model

torch.manual_seed(3)
TORCH_MINIMUM_VERSION = "1.8.0"


def digit_version(version_str: str) -> list[int]:
    """Convert a version string into a list of integers for comparison.

    This function parses version strings with complex formats and converts them into
    comparable numeric arrays. It handles standard version numbers (like '1.2.3')
    as well as release candidates (containing 'rc').

    For standard version components, each number is directly converted to an integer.
    For release candidates (e.g., '2rc1'), the function treats them as slightly
    earlier than the final release by:
    - Converting the number before 'rc' to (number - 1)
    - Appending the rc number as an additional version component

    Examples:
        '1.2.3' -> [1, 2, 3]
        '0.1.2rc1' -> [0, 1, 1, 1]  # 2rc1 becomes [1, 1]
        '2.0rc1' -> [2, -1, 1]  # 0rc1 becomes [-1, 1]

    Args:
        version_str (str): The version string to convert.

    Returns:
        list[int]: A list of integers representing the version for comparison.
    """
    digit_version = []
    for x in version_str.split("."):  # Split the version string by '.'
        if x.isdigit():  # Check if the part is a digit
            digit_version.append(int(x))  # Append the digit as an integer
        elif x.find("rc") != -1:  # Check if the part contains 'rc'
            patch_version = x.split("rc")  # Split the part by 'rc'
            digit_version.append(
                int(patch_version[0]) - 1
            )  # Append the number before 'rc' minus 1
            digit_version.append(int(patch_version[1]))  # Append the number after 'rc'
    return digit_version


def check_torch_version() -> None:
    """Validate that the installed PyTorch version meets the minimum requirement.

    Raises:
        RuntimeError: If the installed PyTorch version is below TORCH_MINIMUM_VERSION.
    """
    torch_version = digit_version(torch.__version__)
    if torch_version < digit_version(TORCH_MINIMUM_VERSION):
        raise RuntimeError(
            f"Torch=={torch.__version__} is not supported for converting to "
            f"torchscript. Please install pytorch>={TORCH_MINIMUM_VERSION}."
        )


def pytorch2torchscript(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
    device: str,
    show_graph: bool = False,
    output_file: str = "tmp.pt",
    verify: bool = False,
) -> None:
    """Export Pytorch model to TorchScript model and verify the outputs are
    same between Pytorch and TorchScript.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        show_graph (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the
            output TorchScript model. Default: `tmp.pt`.
        verify (bool): Whether compare the outputs between
            Pytorch and TorchScript. Default: False.
    """

    # Clear CUDA cache before starting conversion
    if device == "cuda" or device.startswith("cuda:"):
        torch.cuda.empty_cache()
        print(f"Cleared CUDA cache before conversion")

    # replace the original forward with forward_dummy
    # model.forward = model.forward_dummy
    model.eval()

    # Use no_grad context to avoid storing gradients during tracing
    # Create inputs inside the context to minimize memory footprint
    with torch.no_grad():
        inputs = torch.rand(input_shape).to(device)
        traced_model = torch.jit.trace(
            model,
            example_inputs=inputs,
            check_trace=verify,
        )
        # Explicitly delete inputs and clear cache to free memory
        del inputs
        if device == "cuda" or device.startswith("cuda:"):
            torch.cuda.empty_cache()

    if show_graph:
        print(traced_model.graph)

    # Clear CUDA cache before saving to free up memory
    if device == "cuda" or device.startswith("cuda:"):
        torch.cuda.empty_cache()
        print(f"Cleared CUDA cache before saving")

    traced_model.save(output_file)
    print(f"Successfully exported TorchScript model: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .pth checkpoint to TorchScript"
    )
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--checkpoint", help="Checkpoint file")
    parser.add_argument(
        "--show-graph", action="store_true", help="show TorchScript graph"
    )
    parser.add_argument(
        "--verify", action="store_true", help="verify the TorchScript model"
    )
    parser.add_argument("--output-file", type=str, default="tmp.pt")
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1024, 768],
        help="input image size (height, width)",
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    check_torch_version()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    # build the model, load checkpoint
    model = init_model(args.config, args.checkpoint, device=args.device)

    ## create the output directory if it does not exist
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # convert the PyTorch model to TorchScript model
    pytorch2torchscript(
        model,
        input_shape=input_shape,
        device=args.device,
        show_graph=args.show_graph,
        output_file=args.output_file,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
