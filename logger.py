import logging
import sys


def config_logging(level: str, filename, no_tty=False):
    # level = getattr(logging, level)
    # Create a StreamHandler and set its level and format
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)  # Set the desired level for the console
    stream_formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)d %(module)s.%(funcName)s: %(levelname)-8s %(message)s')
    stream_handler.setFormatter(stream_formatter)

    # Create a FileHandler and set its level and format
    file_handler = logging.FileHandler(filename, mode='a')
    file_handler.setLevel(level)  # Set the desired level for the file
    file_formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)d %(module)-12s: %(levelname)-8s %(message)s')
    file_handler.setFormatter(file_formatter)

    handlers = [file_handler]
    if not no_tty:
        handlers.append(stream_handler)

    print(handlers)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=level,
        handlers=handlers)
    effLevel = logging.getLogger().getEffectiveLevel()
    logging.error(f"Logger set to: {effLevel}")
    print(f"Logger set to: {effLevel}")
    return logger
