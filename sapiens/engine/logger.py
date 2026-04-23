# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from datetime import datetime
from logging import Logger as BaseLogger, LogRecord
from pathlib import Path
from typing import Optional, Union

from sapiens.registry import LOGGERS
from termcolor import colored


class FilterDuplicateWarning(logging.Filter):
    def __init__(self, name: str = "sapiens"):
        super().__init__(name)
        self.seen: set = set()

    def filter(self, record: LogRecord) -> bool:
        """Filter the repeated warning message."""
        if record.levelno != logging.WARNING:
            return True

        if record.msg not in self.seen:
            self.seen.add(record.msg)
            return True
        return False


class Formatter(logging.Formatter):
    """Colorful format forLogger."""

    _color_mapping: dict = dict(
        ERROR="red", WARNING="yellow", INFO="white", DEBUG="green"
    )

    def __init__(self, color: bool = True, blink: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert not (not color and blink), (
            "blink should only be available when color is True"
        )

        # Get prefix format according to color.
        error_prefix = self._get_prefix("ERROR", color, blink=True)
        warn_prefix = self._get_prefix("WARNING", color, blink=True)
        info_prefix = self._get_prefix("INFO", color, blink)
        debug_prefix = self._get_prefix("DEBUG", color, blink)

        # Config output format.
        self.err_format = (
            f"%(asctime)s - %(name)s - {error_prefix} - "
            "%(pathname)s - %(funcName)s - %(lineno)d - "
            "%(message)s"
        )
        self.warn_format = f"%(asctime)s - %(name)s - {warn_prefix} - %(message)s"
        self.info_format = f"%(asctime)s - %(name)s - {info_prefix} - %(message)s"
        self.debug_format = f"%(asctime)s - %(name)s - {debug_prefix} - %(message)s"

    def _get_prefix(self, level: str, color: bool, blink=False) -> str:
        """Get the prefix of the target log level."""
        if color:
            attrs = ["underline"]
            if blink:
                attrs.append("blink")
            prefix = colored(level, self._color_mapping[level], attrs=attrs)
        else:
            prefix = level
        return prefix

    def format(self, record: LogRecord) -> str:
        """Override the logging.Formatter.format method."""
        if record.levelno == logging.ERROR:
            self._style._fmt = self.err_format
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warn_format
        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_format
        elif record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_format

        result = logging.Formatter.format(self, record)
        return result


@LOGGERS.register_module()
class Logger(BaseLogger):
    """Standalone logger."""

    _instances = {}

    def __init__(
        self,
        name: str = "sapiens",
        logger_name: str = "sapiens",
        log_file: Optional[str] = None,
        log_level: Union[int, str] = "INFO",
        file_mode: str = "w",
        log_interval: int = 10,
        dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        super().__init__(logger_name)

        self._name = name
        self._log_interval = log_interval
        time_log_file = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._log_file = log_file or (
            Path(dir).mkdir(parents=True, exist_ok=True)
            or Path(dir) / time_log_file / f"{time_log_file}.log"
            if dir
            else None
        )
        # pyre-ignore
        self._log_dir = os.path.dirname(self._log_file) if self._log_file else None

        log_level = (
            logging._nameToLevel[log_level] if isinstance(log_level, str) else log_level
        )

        self._add_stream_handler(log_level, logger_name)
        self._add_file_handler(log_level, logger_name, file_mode)

        Logger._instances[name] = self

    def _add_stream_handler(self, level: int, logger_name: str):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(Formatter(color=True, datefmt="%m/%d %H:%M:%S"))
        handler.addFilter(FilterDuplicateWarning(logger_name))
        self.addHandler(handler)

    def _add_file_handler(self, level: int, logger_name: str, mode: str):
        if self._log_file is None:
            return
        os.makedirs(os.path.dirname(self._log_file), exist_ok=True)
        handler = logging.FileHandler(self._log_file, mode)
        handler.setLevel(level)
        handler.setFormatter(Formatter(color=False, datefmt="%Y/%m/%d %H:%M:%S"))
        handler.addFilter(FilterDuplicateWarning(logger_name))
        self.addHandler(handler)

    @property
    def log_file(self):
        return self._log_file

    @classmethod
    def get_instance(cls, name: str, **kwargs) -> "Logger":
        """Get or create a logger instance."""
        if name not in cls._instances:
            cls._instances[name] = cls(name, **kwargs)
        return cls._instances[name]

    @classmethod
    def get_current_instance(cls) -> "Logger":
        """Get the most recently created logger instance."""
        if not cls._instances:
            cls.get_instance("lca")
        return list(cls._instances.values())[-1]


def print_log(
    msg, logger: Optional[Union[Logger, str]] = None, level=logging.INFO
) -> None:
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif logger == "current":
        logger_instance = Logger.get_current_instance()
        logger_instance.log(level, msg)
    elif isinstance(logger, str):
        if logger in Logger._instances:
            logger_instance = Logger.get_instance(logger)
            logger_instance.log(level, msg)
        else:
            raise ValueError(f"Logger: {logger} has not been created!")
    else:
        raise TypeError(
            "`logger` should be either a logging.Logger object, str, "
            f'"silent", "current" or None, but got {type(logger)}'
        )
