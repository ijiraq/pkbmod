import logging
import sys


def config_logging(level: str, filename, no_tty=False):
    """Configure root logger: file at *level*; stderr at max(INFO, level) unless *no_tty*."""
    file_level = getattr(logging, level.upper(), logging.INFO)
    stream_level = max(logging.INFO, file_level)

    stream_formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)d %(module)s.%(funcName)s: %(levelname)-8s %(message)s"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)d %(module)-12s: %(levelname)-8s %(message)s"
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(filename, mode="a", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    root.addHandler(file_handler)

    if not no_tty:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(stream_level)
        stream_handler.setFormatter(stream_formatter)
        root.addHandler(stream_handler)

    return logging.getLogger(__name__)
