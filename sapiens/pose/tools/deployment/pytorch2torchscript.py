# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# originally copied from https://www.internalfb.com/code/fbsource/[671aa4920700]/fbcode/xrcia/projects/sapiens/experimental_ghe_import/sapiens2/sapiens/seg/tools/deployment/pytorch2torchscript.py?lines=1-204

import argparse
import os

import torch
import torch._C
import torch.serialization
from sapiens.dense.tools.deployment.pytorch2torchscript import check_torch_version
from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
from sapiens.pose.models import init_model

torch.manual_seed(3)
TORCH_MINIMUM_VERSION = "1.8.0"


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

    inputs = torch.rand(input_shape).to(device)

    # replace the original forward with forward_dummy
    # model.forward = model.forward_dummy
    model.eval()
    traced_model = torch.jit.trace(
        model,
        example_inputs=inputs,
        check_trace=verify,
    )

    if show_graph:
        print(traced_model.graph)

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

    ## add pose metainfo to model
    num_keypoints = model.cfg.num_keypoints
    if num_keypoints == 308:
        model.pose_metainfo = parse_pose_metainfo(
            dict(from_file="configs/_base_/keypoints308.py")
        )

    ## add codec to model
    codec_type = model.cfg.codec.pop("type")
    assert codec_type == "UDPHeatmap", "Only support UDPHeatmap"
    model.codec = UDPHeatmap(**model.cfg.codec)

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
