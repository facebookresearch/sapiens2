# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from sapiens.engine.runners import BaseRunner


## left-right flip for pose val and test
class PoseRunner(BaseRunner):
    def test(self) -> None:
        if self.accelerator.is_main_process:
            self.logger.info(f"\033[95mStarting test...\033[0m")

        self.model.eval()
        self.evaluator.reset()
        for i, data_batch in enumerate(self.val_dataloader):
            data_batch = self.data_preprocessor(data_batch)  # preprocess
            inputs, data_samples = data_batch["inputs"], data_batch["data_samples"]

            with torch.no_grad():
                pred = self.model(inputs)  # forward

            if self.val_cfg.get("flip_test", False):
                with torch.no_grad():
                    pred_flipped = self.model(inputs.flip(-1))  # forward

                flip_indices = data_samples[0]["meta"]["flip_indices"]
                pred_flipped = pred_flipped.flip(-1)  ## B x K x heatmap_H x heatmap_W
                assert len(flip_indices) == pred_flipped.shape[1]  ## K
                pred_flipped = pred_flipped[:, flip_indices]
                pred = (pred + pred_flipped) / 2.0

            if self.accelerator.is_main_process and i > 0 and i % 100 == 0:
                self.logger.info(
                    f"\033[95mTest: {i}/{len(self.val_dataloader)}: batch_size: {len(data_batch['inputs'])}\033[0m"
                )
            self.evaluator.process(
                pred, data_samples, accelerator=self.accelerator
            )  ## accelerator used to gather and dedup in val

        # metrics eval on main process
        metrics = self.evaluator.evaluate(
            logger=self.logger, accelerator=self.accelerator
        )
        if self.accelerator.is_main_process:
            self.logger.info(
                f"\033[95mTest: {', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])}\033[0m"
            )
            self.logger.info(f"\033[95mTesting finished ✔\033[0m")

    # -------------------------------------------------------------------------
    def val(self) -> None:
        self.model.eval()

        if self.accelerator.is_main_process:
            self.logger.info(f"\033[95mValidating iter {self.iter}\033[0m")

        self.evaluator.reset()
        for i, data_batch in enumerate(self.val_dataloader):
            data_batch = self.data_preprocessor(data_batch)  # preprocess
            inputs, data_samples = data_batch["inputs"], data_batch["data_samples"]

            with torch.no_grad():
                pred = self.model(inputs)  # forward

            if self.val_cfg.get("flip_test", False):
                with torch.no_grad():
                    pred_flipped = self.model(inputs.flip(-1))  # forward

                flip_indices = data_samples[0]["meta"]["flip_indices"]
                pred_flipped = pred_flipped.flip(-1)  ## B x K x heatmap_H x heatmap_W
                assert len(flip_indices) == pred_flipped.shape[1]  ## K
                pred_flipped = pred_flipped[:, flip_indices]
                pred = (pred + pred_flipped) / 2.0

            if self.accelerator.is_main_process and i > 0 and i % 100 == 0:
                self.logger.info(
                    f"\033[95mVal: {i}/{len(self.val_dataloader)}: batch_size: {len(data_batch['inputs'])}\033[0m"
                )

            self.evaluator.process(pred, data_samples, accelerator=self.accelerator)
        metric = self.evaluator.evaluate(
            logger=self.logger, accelerator=self.accelerator
        )
        self.model.train()
        return metric
