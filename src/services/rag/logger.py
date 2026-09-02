import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"RAG-{name}")