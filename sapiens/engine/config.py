# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import ast
import json
from importlib import util as _iu
from pathlib import Path
from types import ModuleType, SimpleNamespace


class Config(SimpleNamespace):
    """Dot-dict wrapper around a Python-module config."""

    DELETE_KEY = "_delete_"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # keep SimpleNamespace behaviour
        object.__setattr__(
            self,
            "_cfg_dict",  # use object.__setattr__ to avoid
            json.loads(json.dumps(kwargs)),
        )  # recursion hook

    @classmethod
    def fromfile(cls, path: str | Path) -> "Config":
        path = Path(path).expanduser()
        spec = _iu.spec_from_file_location(path.stem, path)
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        data = {
            k: v
            for k, v in vars(mod).items()
            if not k.startswith("_") and not isinstance(v, ModuleType)
        }
        data["filename"] = str(path)
        return cls(**data)

    def merge_from_dict(
        self, options: dict[str, object], allow_list_keys: bool = True
    ) -> None:
        """Merge a dict of dotted keys (or nested dict) into this Config."""
        # Lazily ensure _cfg_dict exists if __init__ was bypassed
        if not hasattr(self, "_cfg_dict"):
            object.__setattr__(self, "_cfg_dict", self.to_dict())

        # 1. expand dotted keys -> nested dict
        nested_patch: dict = {}
        for dotted_key, value in options.items():
            node = nested_patch
            parts = dotted_key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value

        # 2. deep-merge
        merged = self._merge_a_into_b(
            nested_patch, self._cfg_dict, allow_list_keys=allow_list_keys
        )
        object.__setattr__(self, "_cfg_dict", merged)

        # 3. rebuild attributes so dot-access sees the new values
        self.__dict__.update(merged)

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_a_into_b(a: dict, b: dict | list, *, allow_list_keys: bool = False):
        """Deep-merge *a* into *b* (returns a **new** structure)."""
        from copy import deepcopy

        # If the target is a list and we allow list keys ---------------
        if allow_list_keys and isinstance(b, list):
            b = deepcopy(b)
            for k, v in a.items():
                if not k.isdigit():
                    raise TypeError(f"Expected int-like key for list merge, got {k!r}")
                idx = int(k)
                if idx >= len(b):
                    raise IndexError(f"Index {idx} out of range for list.")
                b[idx] = Config._merge_a_into_b(
                    v, b[idx], allow_list_keys=allow_list_keys
                )
            return b

        # Otherwise we are merging dicts -------------------------------
        b = deepcopy(b)
        for k, v in a.items():
            # Deletion flag (_delete_ = True) --------------------------
            if isinstance(v, dict) and v.pop(Config.DELETE_KEY, False):
                b[k] = Config._merge_a_into_b(v, {}, allow_list_keys)
                continue

            if (
                k in b
                and isinstance(b[k], (dict, list))
                and isinstance(v, (dict, list))
            ):
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys=allow_list_keys)
            else:
                b[k] = deepcopy(v)
        return b

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def get(self, key, default=None):
        """Get a value by key with an optional default if key doesn't exist."""
        return self.__dict__.get(key, default)

    def to_dict(self) -> dict:
        def _rec(obj):
            if isinstance(obj, Config):
                return {k: _rec(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, dict):
                return {k: _rec(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_rec(x) for x in obj)
            return obj

        return _rec(self)


# ---------------------------------------------------------------------------
class DictAction(argparse.Action):
    def __call__(self, _parser, namespace, values, _option_string=None):
        out: dict[str, object] = {}
        for kv in values:
            if "=" not in kv:
                raise ValueError(f"--cfg-options expected key=value, got {kv}")
            key, val = kv.split("=", 1)
            for parser_fn in (json.loads, ast.literal_eval, str):
                try:
                    val = parser_fn(val)
                    break
                except Exception:
                    continue
            cur = out
            *parents, leaf = key.split(".")
            for p in parents:
                cur = cur.setdefault(p, {})
            cur[leaf] = val
        setattr(namespace, self.dest, out)


# ---------------------------------------------------------------------------
_INDENT = 4


def _indent_block(txt: str, n: int = _INDENT) -> str:
    pad = " " * n
    return "\n".join(pad + ln if ln else ln for ln in txt.splitlines())


def _format_basic(k, v, mapping: bool) -> str:
    v_str = repr(v) if isinstance(v, str) else str(v)
    if mapping:
        k_str = repr(k) if isinstance(k, str) else str(k)
        return f"{k_str}: {v_str}"
    return f"{k}={v_str}"


def _format_list_tuple(obj, mapping_key=None, mapping=False):
    left, right = ("[", "]") if isinstance(obj, list) else ("(", ")")
    body = []
    for item in obj:
        if isinstance(item, dict):
            body.append("dict(" + _indent_block(_format_dict(item), _INDENT) + "),")
        elif isinstance(item, (list, tuple)):
            body.append(_indent_block(_format_list_tuple(item), _INDENT) + ",")
        else:
            body.append(repr(item) + ",")
    inner = "\n".join(body)
    if mapping_key is None:
        return _indent_block(left + "\n" + inner + "\n" + right, _INDENT)
    if mapping:
        k_str = repr(mapping_key) if isinstance(mapping_key, str) else str(mapping_key)
        return f"{k_str}: {left}\n{_indent_block(inner, _INDENT)}\n{right}"
    return f"{mapping_key}={left}\n{_indent_block(inner, _INDENT)}\n{right}"


def _contains_non_identifier(keys):
    return any((not str(k).isidentifier()) for k in keys)


def _format_dict(d: dict, outer: bool = False) -> str:
    lines = []
    mapping = _contains_non_identifier(d)
    if mapping and not outer:
        lines.append("{")
    for idx, (k, v) in enumerate(sorted(d.items(), key=lambda x: str(x[0]))):
        is_last = idx == len(d) - 1
        suffix = "" if outer or is_last else ","
        if isinstance(v, dict):
            inner = _format_dict(v)
            if mapping:
                k_str = repr(k) if isinstance(k, str) else str(k)
                line = f"{k_str}: dict(\n{_indent_block(inner, _INDENT)}\n){suffix}"
            else:
                line = f"{k}=dict(\n{_indent_block(inner, _INDENT)}\n){suffix}"
        elif isinstance(v, (list, tuple)):
            line = _format_list_tuple(v, mapping_key=k, mapping=mapping) + suffix
        else:
            line = _format_basic(k, v, mapping) + suffix
        lines.append(_indent_block(line, _INDENT))
    if mapping and not outer:
        lines.append("}")
    return "\n".join(lines)


# -----------------------------------------------------------------
def pretty_text(cfg_dict: dict) -> str:
    body = _format_dict(cfg_dict, outer=True)

    import textwrap

    body = textwrap.dedent(body)

    try:
        from yapf.yapflib.yapf_api import FormatCode

        yapf_style = dict(
            based_on_style="pep8",
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True,
        )
        body, _ = FormatCode(body, style_config=yapf_style)
    except ImportError:
        pass
    except Exception as exc:  # keep raw body if yapf fails
        print(f"[pretty_text] yapf failed: {exc}\nReturning un-formatted text.")
    return body


def print_cfg(cfg_dict):
    try:
        from rich.console import Console
        from rich.syntax import Syntax

        console = Console()
        code_str = pretty_text(cfg_dict)
        syntax_block = Syntax(code_str, "python", theme="ansi_dark", word_wrap=False)
        console.print(syntax_block, style="green")
        return "---"
    except ImportError:
        from pprint import pformat

        GREEN = "\033[92m"
        RESET = "\033[0m"
        print(GREEN + pformat(cfg_dict, sort_dicts=False) + RESET)
        return "---"
