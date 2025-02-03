import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
import sys

# Define a custom theme for the console output (optional)
custom_theme = Theme({
    "logging.level.debug": "cyan",
    "logging.level.info": "green",
    "logging.level.warning": "yellow",
    "logging.level.error": "bold red",
    "logging.level.critical": "bold white on red",
})

# Create a console object with the custom theme
console = Console(theme=custom_theme)

# Set up the logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(message)s",  # Format for the console output
    datefmt="[%X]",  # Date format for the console output
    handlers=[
        RichHandler(
            console=console,
            show_path=True,  # Show the file path and line number
            markup=True,  # Enable rich markup in log messages
        ),  # RichHandler for console output
    ]
)

# Create a logger
logger = logging.getLogger("rich")

# Add a FileHandler with the same formatting as the console
file_handler = logging.FileHandler("app.log")
file_formatter = logging.Formatter(
    fmt="[%(asctime)s] %(levelname)-8s %(message)s [%(filename)s:%(lineno)d]",  # Include filename and line number
    datefmt="%Y-%m-%d %H:%M:%S"  # Customize date format for the file
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setStream(sys.stdout)
        handler.stream.reconfigure(encoding='utf-8')


