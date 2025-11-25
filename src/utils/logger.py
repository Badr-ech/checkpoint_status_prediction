"""
Logger utility for the checkpoint prediction system
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with both file and console handlers
    
    Args:
        name: Name of the logger
        log_file: Optional log file name (will be placed in logs/ directory)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = f"{name}.log"
    
    file_path = log_dir / log_file
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    return logger
