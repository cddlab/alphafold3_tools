import os
import sys

from loguru import logger

os.environ["JUPYTER_PLATFORM_DIRS"] = "1"


def log_setup(level="SUCCESS"):
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green>-{level}: {message}",
        level=level,
    )
