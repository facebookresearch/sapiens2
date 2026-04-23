# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import pkgutil

from .. import __version__

_src = pathlib.Path(__file__).with_name("src")
__path__ = pkgutil.extend_path(__path__, __name__)  # allow namespace merge
__path__.append(str(_src))
del pathlib, pkgutil, _src


# -----------------------------------------------------
from importlib import import_module as _imp

_pkg = _imp(__name__ + ".src")  # runs src/__init__.py
