import logging
import sys
import inspect

# -------------------------------
# Usage Example
# -------------------------------
# Call this once at application startup to configure the logger
# from rotab.utils.logger import configure_logger, get_logger
# configure_logger(level=logging.DEBUG, force=True)
#
# Then retrieve a logger in each module as needed
# logger = get_logger()
# logger.info("Logger initialized.")

_logger_configured = False


def configure_logger(level=logging.INFO, to_file: str = None, force=False):
    global _logger_configured
    if _logger_configured and not force:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    handler = logging.FileHandler(to_file, encoding="utf-8") if to_file else logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    _logger_configured = True


def get_logger() -> logging.Logger:
    frame = inspect.stack()[1]
    name = frame.frame.f_globals.get("__name__", "rotab")
    return logging.getLogger(name)
