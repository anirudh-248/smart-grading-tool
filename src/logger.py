import logging
from logging import Logger
from .config import Config

def get_logger(name: str = "smart_grader") -> Logger:
    cfg = Config()
    level_name = cfg.get('logging', {}).get('level', 'INFO').upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
