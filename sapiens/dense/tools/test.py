# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch

# pyre-ignore[21]: Cannot find module `sapiens.engine.config`
from sapiens.engine.config import Config, DictAction
from sapiens.engine.runners import *


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a segmentor")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args(argv)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint

    # set train to false
    cfg.train_dataloader = None

    # start testing
    runner_type = cfg.get("runner_type", "BaseRunner")
    runner = eval(runner_type).from_cfg(cfg)
    runner.test()


if __name__ == "__main__":
    main()
