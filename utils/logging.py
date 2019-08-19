from typing import List
import os
import sys
import logging
from logging import LogRecord
from datetime import datetime
from collections import OrderedDict

from pythonjsonlogger import jsonlogger


class JsonFormatter(jsonlogger.JsonFormatter):

    def parse(self) -> List[str]:
        """https://docs.python.jp/3/library/logging.html"""
        return ['timestamp', 'level', 'name', 'message']

    def add_fields(
            self, log_record: OrderedDict, record: LogRecord, message_dict: dict) -> None:
        super().add_fields(log_record, record, message_dict)

        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().isoformat()

        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname


loggers = {}


def get_logger(name: str = 'object-detection-template'):
    if name in loggers:
        return loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    loggers[name] = logger

    return logger
