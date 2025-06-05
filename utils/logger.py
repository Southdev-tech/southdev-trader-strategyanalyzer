import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


def log_info(message):
    logging.info(message)

def log_error(message):
    logging.error(message)

def log_warning(message):
    logging.warning(message)

def log_debug(message):
    logging.debug(message)