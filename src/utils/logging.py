# logging_config.py
import logging
from pathlib import Path
from datetime import datetime

_LOG_FILE = None

def setup_logging():
    """Call this once at application startup."""
    global _LOG_FILE
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE = logs_dir / f"{timestamp}.log"
    
    return _LOG_FILE

def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # File handler
        file_handler = logging.FileHandler(_LOG_FILE)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger