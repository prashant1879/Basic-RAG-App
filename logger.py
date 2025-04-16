import logging
import asyncio
from logging.handlers import RotatingFileHandler
import os

# Create logs directory if not exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configure the logger
logger = logging.getLogger("async-safe-logger")
logger.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler
file_handler = RotatingFileHandler("logs/async_app.log", maxBytes=1_000_000, backupCount=3)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
