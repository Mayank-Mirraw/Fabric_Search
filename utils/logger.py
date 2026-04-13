# utils/logger.py
import sys
from loguru import logger
import os

# Create a folder for logs if it doesn't exist
os.makedirs("logs", exist_ok=True)

def get_logger(name: str):
    logger.remove() # Clean up default settings
    
    # 1. Show nice colored messages in your VS Code terminal
    logger.add(sys.stdout, colorize=True, 
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}", 
               level="INFO")
    
    # 2. Save the details to a file so you can check errors later
    logger.add("logs/fabric_search.log", rotation="10 MB", 
               format="{time} | {level} | {name} | {message}", 
               level="DEBUG")
    
    return logger.bind(name=name)