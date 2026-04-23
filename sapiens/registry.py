# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Callable, Dict, Type, Union


class Registry(dict):
    def register(
        self, obj: Union[Type, Callable] | None = None, *, name: str | None = None
    ):
        def _do_register(o):
            key = name or o.__name__
            if key in self:
                return self[key]  # Skip registration if already exists
            self[key] = o
            return o

        return _do_register(obj) if obj is not None else _do_register

    def register_module(self, *args, name: str | None = None):
        if args and callable(args[0]):
            return self.register(args[0], name=name)

        def decorator(obj):
            return self.register(obj, name=name)

        return decorator

    def build(self, cfg: Dict[str, Any], **extra_kwargs) -> Any:
        cfg = dict(cfg)  # shallow copy
        obj_type = cfg.pop("type")
        if obj_type not in self:
            raise KeyError(f"{obj_type!r} not found in registry.")
        cls_or_fn = self[obj_type]
        return cls_or_fn(**cfg, **extra_kwargs)


# --------------------------------------------------------------------------- #
MODELS = Registry()
DATASETS = Registry()
TRANSFORMS = Registry()
OPTIMIZERS = Registry()
SCHEDULERS = Registry()
LOGGERS = Registry()
VISUALIZERS = Registry()
HOOKS = Registry()

__all__ = [
    "Registry",
    "MODELS",
    "DATASETS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "LOGGERS",
    "VISUALIZERS",
    "HOOKS",
]
