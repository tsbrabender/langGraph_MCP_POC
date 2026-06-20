"""Structured logging setup using structlog."""

import logging
import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog with JSON-style structured output.

    Args:
        log_level: Standard Python log level string (e.g. "INFO", "DEBUG").
    """
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structured logger.

    Args:
        name: Logger name, typically __name__ of the calling module.
    """
    return structlog.get_logger(name)
