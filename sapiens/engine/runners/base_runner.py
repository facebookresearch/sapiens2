# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import random
import reprlib
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.parallelism_config import ParallelismConfig
from accelerate.utils import (
    DistributedDataParallelKwargs,
    FullyShardedDataParallelPlugin,
    TorchDynamoPlugin,
)
from safetensors.torch import load_file
from sapiens.registry import (
    DATASETS,
    LOGGERS,
    MODELS,
    OPTIMIZERS,
    SCHEDULERS,
    VISUALIZERS,
)
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader

from ..config import pretty_text

_repr = reprlib.Repr()
_repr.maxlist = 10


# ---------------------------------------------------------------------------
class BaseRunner:
    def __init__(
        self,
        *,
        model: dict | nn.Module,
        work_dir: str,
        train_dataloader: dict | DataLoader | None = None,
        val_dataloader: dict | None = None,
        val_cfg: dict | None = None,
        data_preprocessor: dict | None = None,
        accelerator_cfg: Dict[str, Any],
        optimizer: dict | torch.optim.Optimizer,
        scheduler: dict | None = None,
        clip_grad: Dict[str, Any] | None = None,
        logger: dict | None = None,
        checkpoint: dict | None = None,
        visualizer: dict | None = None,
        randomness: Dict[str, Any] | None = None,
        cfg: Dict[str, Any] | None = None,
        **_ignored,
    ) -> None:
        self.cfg = cfg
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._init_env()
        self._set_seed(randomness or {})
        self._init_logger(logger=logger)
        self._log_config()
        self._init_accelerator(accelerator_cfg)

        # train dataloader
        self.train_dataloader = None
        if train_dataloader is not None:
            train_dataset = DATASETS.build(train_dataloader["dataset"])
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=train_dataloader.get("batch_size", 1),
                shuffle=train_dataloader.get("shuffle", True),
                num_workers=train_dataloader.get("num_workers", 0),
                persistent_workers=train_dataloader.get("persistent_workers", True),
                pin_memory=train_dataloader.get("pin_memory", True),
            )

        # val dataloader
        self.val_dataloader = None
        if val_dataloader is not None and val_cfg is not None:
            val_dataset = DATASETS.build(val_dataloader["dataset"])
            collate_fn_cfg = val_dataloader.get("collate_fn")
            collate_fn_obj = (
                MODELS.get(collate_fn_cfg["type"]) if collate_fn_cfg else None
            )
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=val_dataloader.get("batch_size", 1),
                shuffle=val_dataloader.get("shuffle", False),
                num_workers=val_dataloader.get("num_workers", 0),
                persistent_workers=val_dataloader.get("persistent_workers", True),
                pin_memory=val_dataloader.get("pin_memory", True),
                collate_fn=collate_fn_obj,
                multiprocessing_context=val_dataloader.get(
                    "multiprocessing_context", None
                ),
            )
            self.val_cfg = val_cfg
            self.val_every = self.val_cfg.get("val_interval", 100)
            self.evaluator = MODELS.build(self.val_cfg["evaluator"])

        self.data_preprocessor = MODELS.build(data_preprocessor)  # data_preprocessor
        self.model = MODELS.build(model)

        # optimizer, scheduler, clip_grad
        self.optimizer = self._build_optimizer(optimizer)
        self.scheduler = SCHEDULERS.build(scheduler, optimizer=self.optimizer)
        self.clip_grad = clip_grad  # clip_grad

        self.visualizer = None
        if self.train_dataloader is not None:
            self.visualizer = (
                VISUALIZERS.build(
                    {**visualizer, "output_dir": self.work_dir / "vis_data"}
                )
                if visualizer
                else None
            )

        # prepare
        self._prepare_accelerator()
        self._print_model()

        ## logging params
        self.log_every = self.logger._log_interval if self.logger else 0
        self.save_every = (checkpoint or {}).get("save_interval", 0)
        self.vis_every = self.visualizer.vis_interval if self.visualizer else 0

    # --------------------------------------------------------------------------
    def train(self) -> None:
        self.model.train()

        data_iter = iter(self.train_dataloader)

        while self.iter < self.max_iters:
            t = time.time()

            if not self.gpu_profiler_disabled:
                self.gpu_profiler.before_step()

            try:
                data_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                data_batch = next(data_iter)
            data_time = time.time() - t

            # ------------------------------------------------------
            with self.accelerator.autocast(), self.accelerator.accumulate(self.model):
                t = time.time()

                loss, logs = self.forward(data_batch)
                self.accelerator.backward(loss)  # backward

                # step
                grad_norm = self._clip_gradients()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                iter_time = time.time() - t

            # ------------------------------------------------------
            self.iter += 1

            if not self.gpu_profiler_disabled:
                self.gpu_profiler.after_step()

            # ------------------------------------------------------
            if self.save_every and self.iter % self.save_every == 0 and self.iter > 0:
                self._save_checkpoint(f"iter_{self.iter}")

            # ------------------------------------------------------
            if (
                self.visualizer
                and self.iter % self.vis_every == 0
                and self.accelerator.is_main_process
            ):
                self.visualizer.add_batch(data_batch, logs, step=self.iter)
                self.logger.info(f"\033[96mVisualized iter {self.iter}\033[0m")

            # ------------------------------------------------------
            if self.val_dataloader is not None and self.iter % self.val_every == 0:
                val_metrics = self.val()
                logs["val_metrics"] = val_metrics

            if self.accelerator.is_main_process:
                self._log_iter(
                    logs=logs,
                    iter_time=iter_time,
                    data_time=data_time,
                    grad_norm=grad_norm,
                )

        # -------------------------------------------------
        self._save_checkpoint("final")

        self.accelerator.save_model(self.model, self.work_dir / "checkpoints")
        self.accelerator.end_training()

        if self.accelerator.is_main_process:
            self.logger.info("\033[92mTraining finished ✔\033[0m")

    # -------------------------------------------------------------------------
    def forward(self, data_batch: dict) -> tuple[float, dict]:
        data_batch = self.data_preprocessor(data_batch)  # preprocess
        inputs, data_samples = data_batch["inputs"], data_batch["data_samples"]

        if self.pc is not None:
            pred = self.model(inputs, cp=self.accelerator.maybe_context_parallel)
        else:
            pred = self.model(inputs)  # forward

        loss, logs = self.raw_model.loss(pred, data_samples)
        return loss, logs

    # -------------------------------------------------------------------------
    def test(self) -> None:
        if self.accelerator.is_main_process:
            self.logger.info(f"\033[95mStarting test...\033[0m")

        self.model.eval()
        self.evaluator.reset()
        for i, data_batch in enumerate(self.val_dataloader):
            data_batch = self.data_preprocessor(data_batch)  # preprocess
            inputs, data_samples = data_batch["inputs"], data_batch["data_samples"]

            with torch.no_grad():
                if self.pc is not None:
                    pred = self.model(
                        inputs, cp=self.accelerator.maybe_context_parallel
                    )
                else:
                    pred = self.model(inputs)  # forward

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
                if self.pc is not None:
                    pred = self.model(
                        inputs, cp=self.accelerator.maybe_context_parallel
                    )
                else:
                    pred = self.model(inputs)  # forward

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

    # --------------------------------------------------------------------------
    def _clip_gradients(self) -> float | None:
        if not self.clip_grad or not self.accelerator.sync_gradients:
            return None

        max_norm = float(self.clip_grad.get("max_norm", 1.0))
        norm_type = float(self.clip_grad.get("norm_type", 2.0))
        total_norm = self.accelerator.clip_grad_norm_(
            self.model.parameters(), max_norm, norm_type
        )

        return total_norm

    def _log_iter(self, *, logs, iter_time, data_time, grad_norm=None):
        """Call once per iteration; prints every `self._log_every` steps."""
        log_payload = {}
        if "val_metrics" in logs:
            val_metrics = logs.pop("val_metrics")
            log_payload.update(val_metrics)
            self.logger.info(
                f"\033[95mVal-Iter[{self.iter}]: {', '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items()])}\033[0m"
            )

        ## aggregate losses and metrics
        for key in logs:
            if key.startswith("loss_") or key.startswith("acc_"):
                self._loss_acc[key] += float(logs[key].item())

        self._time_acc += iter_time
        self._data_acc += data_time

        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()

        if grad_norm is not None:
            self._grad_acc += float(grad_norm)

        # log every `self._log_every` steps
        if (
            self.log_every > 0
            and (self.iter % self.log_every == 0 or self.iter == self.max_iters - 1)
            and self.iter > 0
        ):
            k = self.log_every
            avg_losses = {
                key: val / k
                for key, val in self._loss_acc.items()
                if key.startswith("loss_")
            }
            total_avg_loss = sum(avg_losses.values())
            avg_time = self._time_acc / k
            avg_data_time = self._data_acc / k
            avg_grad = self._grad_acc / k if self._grad_acc else 0.0

            eta_secs = avg_time * (self.max_iters - self.iter)
            eta = str(datetime.timedelta(seconds=int(eta_secs)))
            mem_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024)

            loss_str_parts = [f"{key}: {val:.4f}" for key, val in avg_losses.items()]
            loss_str = f"loss: {total_avg_loss:.4f}  {'  '.join(loss_str_parts)}"

            acc_str = ""
            for key, val in self._loss_acc.items():
                if key.startswith("acc_"):
                    acc_str += f"{key}: {val / k:.4f}  "
            if acc_str:
                loss_str += f"  {acc_str}"

            if (
                self.optimizer.param_groups[0]["lr"]
                != self.optimizer.param_groups[-1]["lr"]
            ):
                decayed_lr = self.optimizer.param_groups[0]["lr"]
                lr = self.optimizer.param_groups[-1]["lr"]

                lr_str = f"lr: {lr:.2e}  decay_lr: {decayed_lr:.2e}"
            else:
                lr_str = f"lr: {self.optimizer.param_groups[0]['lr']:.2e}"

            self.logger.info(
                f"Iter(train) [{self.iter}/{self.max_iters}]: "
                f"{lr_str}  "
                f"eta: {eta}  "
                f"data_time: {avg_data_time:.2f}  "
                f"iter_time: {avg_time:.2f}  "
                f"memory: {mem_mb}  "
                f"grad_norm: {avg_grad:.2f}  "
                f"{loss_str}"
            )

            log_payload.update(
                {
                    "loss": total_avg_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "grad_norm": avg_grad,
                    "iter_time": avg_time,
                    "data_time": avg_data_time,
                    **avg_losses,  # Add individual average losses
                }
            )

            self.accelerator.log(log_payload, step=self.iter)
            self._loss_acc.clear()
            self._time_acc = self._data_acc = self._grad_acc = 0.0

    # --------------------------------------------------------------------------
    def _save_checkpoint(self, tag: str) -> None:
        checkpoint_dir = self.work_dir / "checkpoints" / tag
        self.accelerator.save_state(output_dir=checkpoint_dir)

        if self.accelerator.is_main_process:
            self.logger.info(
                f"\033[92mCheckpoint saved ➜ {os.path.basename(checkpoint_dir)}\033[0m"
            )

    # --------------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """
        Custom state to be saved by Accelerator.
        """
        return {"iter": torch.tensor(self.iter, dtype=torch.int64, device="cpu")}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load custom state saved by Accelerator.
        """
        self.iter = int(state_dict["iter"])

    def _init_env(self):
        """Setup distributed environment variables if not already set."""
        if "RANK" not in os.environ:
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = f"127.0.0.{random.randint(1, 255)}"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(random.randint(1024, 65535))

    def _init_accelerator(self, accelerator_cfg) -> None:
        """Initialize Accelerator."""
        self.accelerator_cfg = accelerator_cfg.copy()

        compile_cfg = accelerator_cfg.pop("compile_cfg", {})
        dynamo_plugin = TorchDynamoPlugin(**compile_cfg) if compile_cfg else None

        self.dist_type = accelerator_cfg.pop("type", "DDP").upper()  # "DDP" | "FSDP"
        fsdp_cfg = accelerator_cfg.pop("fsdp_cfg", {})
        parallelism_cfg = accelerator_cfg.pop("parallelism_cfg", {})
        self.max_iters = int(accelerator_cfg.pop("max_interval", 1e4))
        self.pc = None

        find_unused_parameters = bool(
            accelerator_cfg.pop("find_unused_parameters", False)
        )

        common_kwargs = dict(
            project_dir=self.work_dir,
            dynamo_plugin=dynamo_plugin,
            **accelerator_cfg,
        )

        if self.dist_type == "FSDP":
            policy_name = fsdp_cfg.pop("auto_wrap_policy", "none")
            min_params = fsdp_cfg.pop("auto_wrap_min_num_params", 1e6)

            if policy_name == "size_based":
                fsdp_cfg["min_num_params"] = min_params
            elif policy_name == "transformer":
                fsdp_cfg["auto_wrap_policy"] = transformer_auto_wrap_policy

            mp_cfg = fsdp_cfg.pop("mixed_precision", None)

            if mp_cfg:
                _DTYPE = {
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                    "fp32": torch.float32,
                }
                fsdp_cfg["mixed_precision_policy"] = MixedPrecisionPolicy(
                    param_dtype=_DTYPE.get(mp_cfg.get("param_dtype", "fp32")),
                    reduce_dtype=_DTYPE.get(mp_cfg.get("reduce_dtype", "fp32")),
                )
            fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_cfg)

            # https://docs.axolotl.ai/docs/nd_parallelism.html
            self.pc = (
                ParallelismConfig(
                    **parallelism_cfg,
                )
                if parallelism_cfg
                else None
            )

            self.accelerator = Accelerator(
                parallelism_config=self.pc, fsdp_plugin=fsdp_plugin, **common_kwargs
            )

        else:  # DDP (default)
            if find_unused_parameters:
                common_kwargs["kwargs_handlers"] = [
                    DistributedDataParallelKwargs(find_unused_parameters=True)
                ]
            self.accelerator = Accelerator(**common_kwargs)

        if self.logger is not None:
            self.accelerator.init_trackers(self.logger._log_dir)

    def _prepare_accelerator(self) -> None:
        self.iter = 0
        self._loss_acc = defaultdict(float)
        self._time_acc = self._data_acc = self._grad_acc = 0.0

        self.accelerator.register_for_checkpointing(self)

        load_from = self.cfg.get("load_from", None)  # path or None
        resume = self.cfg.get("resume", False)

        if load_from and not resume:
            self._load_checkpoint(load_from)

        ## train + val
        if self.train_dataloader is not None and self.val_dataloader is not None:
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.scheduler,
                self.val_dataloader,
                self.evaluator,
            ) = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.scheduler,
                self.val_dataloader,
                self.evaluator,
            )
        ## train only
        elif self.train_dataloader is not None and self.val_dataloader is None:
            self.model, self.optimizer, self.train_dataloader, self.scheduler = (
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_dataloader, self.scheduler
                )
            )
        ## val only
        elif self.train_dataloader is None and self.val_dataloader is not None:
            (
                self.model,
                self.optimizer,
                self.scheduler,
                self.val_dataloader,
                self.evaluator,
            ) = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.scheduler,
                self.val_dataloader,
                self.evaluator,
            )

        ## data_preprocessor
        if self.data_preprocessor is not None:
            model_dtype = None
            if self.accelerator.mixed_precision == "fp16":
                model_dtype = torch.float16
            elif self.accelerator.mixed_precision == "bf16":
                model_dtype = torch.bfloat16

            # Move to device and cast dtype simultaneously.
            self.data_preprocessor = self.data_preprocessor.to(
                device=self.accelerator.device, dtype=model_dtype
            )

        if load_from and resume:
            self._resume(load_from)

        gradient_accumulation_steps = self.accelerator.gradient_accumulation_steps

        if gradient_accumulation_steps > 1:
            self.logger.warning(
                f"Gradient accumulation with {gradient_accumulation_steps} steps is not supported. "
                "LR schedule will be off from expected."
            )

        self.raw_model = self.accelerator.unwrap_model(self.model)

    # --------------------------------------------------------------------------
    def _build_optimizer(self, optimizer):
        optimizer_cfg = optimizer.copy()
        paramwise_cfg = optimizer_cfg.pop("paramwise_cfg", None)

        if paramwise_cfg:
            # Add base lr and weight_decay for the helper to use
            paramwise_cfg["lr"] = optimizer_cfg.get("lr")
            paramwise_cfg["weight_decay"] = optimizer_cfg.get("weight_decay")
            params = self._generate_param_groups(paramwise_cfg)

            if "weight_decay" in optimizer_cfg:
                optimizer_cfg["weight_decay"] = float(
                    optimizer_cfg["weight_decay"] or 0.0
                )

            optimizer_cls = OPTIMIZERS.get(optimizer_cfg.pop("type"))
            return optimizer_cls(params, **optimizer_cfg)
        else:
            return OPTIMIZERS.build(optimizer, params=self.model.parameters())

    def _get_layer_id_for_sapiens(self, var_name: str, num_max_layer: int) -> int:
        """Assigns a layer ID to each parameter for layer-wise decay."""
        # remove fsdp prefix
        if "_fsdp_wrapped_module" in var_name:
            var_name = var_name.replace("_fsdp_wrapped_module.", "")

        if var_name in (
            "backbone.cls_token",
            "backbone.mask_token",
            "backbone.pos_embed",
            "backbone.storage_tokens",
        ):
            return 0
        elif var_name.startswith("backbone.patch_embed"):
            return 0
        elif var_name.startswith("backbone.tokenizer"):
            return 0
        elif var_name.startswith("backbone.layers") or var_name.startswith(
            "backbone.blocks"
        ):
            try:
                # e.g., backbone.layers.10.norm.weight -> 10
                layer_id = int(var_name.split(".")[2])
                return layer_id + 1
            except (ValueError, IndexError):
                # Fallback for unexpected layer name format
                return num_max_layer - 1
        else:
            # All other parameters (e.g., decode_head, final norm) get the highest LR
            return num_max_layer - 1

    def _generate_param_groups(self, paramwise_cfg: dict) -> list:
        """Generates parameter groups using sapiens specific layer decay logic."""
        base_lr = float(paramwise_cfg.get("lr", 0.0))
        base_wd = float(paramwise_cfg.get("weight_decay") or 0.0)

        # Layer decay is optional. If rate==1.0 or num_layers missing -> no layer decay.
        layer_decay_rate = float(paramwise_cfg.get("layer_decay_rate", 1.0))
        num_layers_cfg = paramwise_cfg.get("num_layers")
        use_layer_decay = (layer_decay_rate != 1.0) and (num_layers_cfg is not None)
        if use_layer_decay:
            num_layers = int(num_layers_cfg) + 2

        param_groups = []
        params_map = {}  # Key: (lr, wd) -> list[(name, param)]

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # --- Weight decay per-parameter ---
            if len(param.shape) == 1 or name.endswith(".bias") or "pos_embed" in name:
                this_weight_decay = 0.0
            else:
                this_weight_decay = base_wd

            # --- Learning rate scaling (optional layer-decay) ---
            if use_layer_decay:
                layer_id = self._get_layer_id_for_sapiens(name, num_layers)
                lr_scale = layer_decay_rate ** (num_layers - layer_id - 1)
                this_lr = base_lr * lr_scale
            else:
                this_lr = base_lr

            key = (this_lr, this_weight_decay)
            params_map.setdefault(key, []).append((name, param))

        # materialize groups
        for (lr, wd), named_params in params_map.items():
            params = [p for _, p in named_params]
            param_groups.append({"params": params, "lr": lr, "weight_decay": wd})

        if (
            self.logger
            and self.accelerator.is_main_process
            and self.train_dataloader is not None
        ):
            # Create a new dictionary to group parameters by LR only for logging
            lr_groups = {}
            for (lr, _), named_params in params_map.items():
                if lr not in lr_groups:
                    lr_groups[lr] = []
                lr_groups[lr].extend(named_params)

            log_str = "\033[96mOptimizer parameter groups created:\n"

            # Sort by learning rate and log one line per LR
            for lr, named_params in sorted(lr_groups.items()):
                num_tensors = len(named_params)
                num_params = sum(p.numel() for name, p in named_params)

                param_names = [name for name, p in named_params]
                example_names = ", ".join(param_names[: min(4, len(param_names))])

                if len(param_names) > 4:
                    example_names += ", ..."

                # Use formatting to align columns
                log_str += (
                    f"  - decayed_lr: {lr:<11.4e} | tensors: {num_tensors:<4} | "
                    f"params: {num_params / 1e6:<6.2f}M | names: {example_names}\n"
                )
            log_str += "\033[0m"
            self.logger.info(log_str)

        return param_groups

    # Only loads model weights, not training state. This is to handle the torch.compile preload case.
    def _load_checkpoint(self, load_from: str | os.PathLike):
        load_from = Path(load_from)
        weights_file = None

        if load_from.is_file() and load_from.name.endswith(
            (".safetensors", ".pth", ".bin")
        ):
            weights_file = load_from

        elif load_from.is_dir():
            candidates = ["model.safetensors", "model.pth", "pytorch_model.bin"]
            for name in candidates:
                if (load_from / name).exists():
                    weights_file = load_from / name
                    break

            if not weights_file:
                for d in load_from.glob("*"):
                    if d.is_dir():
                        for name in candidates:
                            if (d / name).exists():
                                weights_file = d / name
                                break
                    if weights_file:
                        break

        if not weights_file or not weights_file.exists():
            raise FileNotFoundError(
                f"Could not find a valid .safetensors, .pth, or .bin file in {load_from}"
            )

        if self.accelerator.is_main_process:
            self.logger.info(f"Loading model weights from: {weights_file}")

        if str(weights_file).endswith(".safetensors"):
            state_dict = load_file(str(weights_file), device="cpu")
        else:  # Handle .pth and .bin files
            checkpoint = torch.load(
                str(weights_file), map_location="cpu", weights_only=False
            )
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

        model_state_dict = self.model.state_dict()
        compatible_state_dict = {}
        mismatched_keys = []

        for key, checkpoint_tensor in state_dict.items():
            if key in model_state_dict:
                model_tensor = model_state_dict[key]

                # Check if the shapes match or if its pos_embed
                if checkpoint_tensor.shape == model_tensor.shape or "pos_embed" in key:
                    compatible_state_dict[key] = checkpoint_tensor
                else:
                    # If shapes do not match, record it and skip loading
                    mismatched_keys.append(
                        f"- {key}: "
                        f"checkpoint has shape {checkpoint_tensor.shape}, "
                        f"model has shape {model_tensor.shape}"
                    )

        incompat = self.model.load_state_dict(compatible_state_dict, strict=False)

        if self.accelerator.is_main_process:
            if mismatched_keys:
                log_str = "\n".join(mismatched_keys)
                self.logger.warning(
                    "\033[31mSize Mismatch (these weights were NOT loaded): \n"
                    f"{log_str}\033[0m"
                )

            if incompat.missing_keys:
                self.logger.warning(
                    "\033[38;5;208mMissing keys (in model, NOT in checkpoint): \n"
                    + "\n".join(incompat.missing_keys)
                    + "\033[0m"
                )
            if incompat.unexpected_keys:
                self.logger.warning(
                    "\033[38;5;208mUnexpected keys (in checkpoint, NOT in model): \n"
                    + "\n".join(_repr.repr(k) for k in incompat.unexpected_keys)
                    + "\033[0m"
                )

            self.logger.info("Model weights loaded successfully ✔")

    def _resume(self, load_from: str | os.PathLike):
        # If a file is provided, use its parent directory as the checkpoint directory
        if str(load_from).endswith((".safetensors", ".pth", ".bin")):
            load_from = Path(load_from).parent

        load_from = str(load_from)

        if self.accelerator.is_main_process:
            self.logger.info(f"Resuming state from: {load_from}")

        self.accelerator.load_state(load_from)

        if self.accelerator.is_main_process:
            self.logger.info("Training state resumed ✔")

    # --------------------------------------------------------------------------
    def _init_logger(self, logger) -> None:
        self.logger = None
        if os.environ.get("RANK", "0") == "0":
            self.logger = LOGGERS.build({**logger, "dir": self.work_dir})

    # --------------------------------------------------------------------------
    def _log_config(self) -> None:
        if os.environ.get("RANK", "0") == "0":
            file = os.path.join(self.work_dir, os.path.basename(self.cfg["filename"]))
            with open(file, "w", encoding="utf-8") as f:
                f.write(pretty_text(self.cfg))
            from pygments import highlight
            from pygments.formatters import TerminalFormatter
            from pygments.lexers import PythonLexer

            self.logger.info(
                highlight(
                    pretty_text(self.cfg),
                    PythonLexer(),
                    TerminalFormatter(style="monokai"),
                )
            )

    # --------------------------------------------------------------------------
    def _set_seed(self, rnd: Dict[str, Any]):
        seed = int(rnd.get("seed", 0))
        deterministic = bool(rnd.get("deterministic", False))
        diff_rank_seed = bool(rnd.get("diff_rank_seed", True))

        rank = 0
        if diff_rank_seed:
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = int(os.environ.get("RANK", "0"))
            seed += rank

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False

    # -------------------------------------------------------------------------
    def _get_model_summary_str(self, model, max_depth=5):
        """Creates a concise, dependency-free summary of a PyTorch model, grouping identical repeating layers."""
        summary_lines = []

        def VRAM_repr(num_params):
            if num_params > 1e9:
                return f"{num_params / 1e9:,.2f}B"
            if num_params > 1e6:
                return f"{num_params / 1e6:,.2f}M"
            if num_params > 1e3:
                return f"{num_params / 1e3:,.2f}K"
            return str(num_params)

        def recurse(module, prefix="", depth=0):
            if depth > max_depth:
                return
            children = list(module.named_children())
            i = 0
            while i < len(children):
                name, child = children[i]
                # Count identical sequential modules
                num_repeats = 1
                for j in range(i + 1, len(children)):
                    next_name, next_child = children[j]
                    if isinstance(next_child, type(child)) and str(next_child) == str(
                        child
                    ):
                        num_repeats += 1
                    else:
                        break

                is_last = (i + num_repeats - 1) == (len(children) - 1)
                connector = "`-- " if is_last else "|-- "
                child_params = sum(p.numel() for p in child.parameters())

                if num_repeats > 1:
                    last_name_in_block = children[i + num_repeats - 1][0]
                    block_name = f"{name}..{last_name_in_block}"
                    total_params = child_params * num_repeats
                    summary_lines.append(
                        f"{prefix}{connector}{block_name} ({type(child).__name__} x {num_repeats}): "
                        f"{VRAM_repr(total_params)} params"
                    )
                else:
                    summary_lines.append(
                        f"{prefix}{connector}{name} ({type(child).__name__}): {VRAM_repr(child_params)} params"
                    )
                    new_prefix = prefix + ("    " if is_last else "|   ")
                    recurse(child, prefix=new_prefix, depth=depth + 1)

                i += num_repeats

        total_params = sum(p.numel() for p in model.parameters())
        summary_lines.append(f"Total params: {VRAM_repr(total_params)}")
        recurse(model)
        return "\n".join(summary_lines)

    def _print_model(self) -> None:
        if not self.logger or not self.accelerator.is_main_process:
            return

        tot, trainable = 0, 0
        for p in self.raw_model.parameters():
            n = p.numel()
            tot += n
            trainable += n if p.requires_grad else 0

        self.logger.info(
            f"\033[92mModel Architecture:\n{self._get_model_summary_str(self.raw_model, max_depth=5)}\033[0m"
        )
        self.logger.info(
            f"\033[92mParameters: {tot / 1e6:.2f} M total | {trainable / 1e6:.2f} M learnable\033[0m"
        )

        if (
            self.accelerator_cfg["type"] == "DDP"
            and "compile_cfg" not in self.accelerator_cfg
        ):
            try:
                from fvcore.nn import FlopCountAnalysis

                dummy_input = torch.randn(
                    1, 3, 1024, 768, device=self.accelerator.device
                )

                flops = FlopCountAnalysis(self.raw_model, dummy_input)
                gflops = flops.total() / 1e9
                self.logger.info(f"\033[92mFLOPs (GMac): {gflops:.2f} GFLOPs\033[0m")
            except Exception as e:
                self.logger.warning(f"Could not calculate FLOPs: {e}")

        if self.train_dataloader is not None:
            unique_lrs = sorted({g["lr"] for g in self.optimizer.param_groups})
            lr_str = ", ".join(f"{v:.4e}" for v in unique_lrs)
            self.logger.info(f"\033[92mInitial Learning Rate(s): {lr_str}\033[0m")

    # --------------------------------------------------------------------------
    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            model=cfg.model,
            work_dir=cfg.work_dir,
            train_dataloader=cfg.train_dataloader,
            val_dataloader=getattr(cfg, "val_dataloader", None),
            val_cfg=getattr(cfg, "val_cfg", None),
            data_preprocessor=cfg.data_preprocessor,
            accelerator_cfg=cfg.accelerator_cfg,
            optimizer=cfg.optimizer,
            scheduler=getattr(cfg, "scheduler", None),
            clip_grad=getattr(cfg, "clip_grad", None),
            logger=getattr(cfg, "logger", None),
            checkpoint=getattr(cfg, "checkpoint", None),
            visualizer=getattr(cfg, "visualizer", None),
            randomness=getattr(cfg, "randomness", None),
            cfg=cfg.to_dict(),
        )
