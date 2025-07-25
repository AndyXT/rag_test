"""Rich console logger utility for consistent output formatting."""
import logging
from rich.console import Console
from typing import Optional


class RichLogger:
    """Centralized logger for consistent Rich console output formatting."""
    
    _console = Console()
    _logger = logging.getLogger("rag_cli")
    debug_mode = False  # Flag for debug mode
    _tui_mode = False  # Flag to disable console output in TUI mode
    
    @classmethod
    def set_tui_mode(cls, enabled: bool) -> None:
        """Enable or disable TUI mode (disables console output to prevent duplicates)."""
        cls._tui_mode = enabled
    
    @classmethod
    def warning(cls, message: str) -> None:
        """Print a warning message in yellow."""
        if not cls._tui_mode:
            cls._console.print(f"[yellow]⚠ {message}[/yellow]")
        cls._logger.warning(message)
    
    @classmethod
    def success(cls, message: str) -> None:
        """Print a success message in green."""
        if not cls._tui_mode:
            cls._console.print(f"[green]✓ {message}[/green]")
        cls._logger.info(f"SUCCESS: {message}")
    
    @classmethod
    def error(cls, message: str, exception: Optional[Exception] = None) -> None:
        """Print an error message in red."""
        if not cls._tui_mode:
            cls._console.print(f"[red]✗ {message}[/red]")
        if exception:
            cls._logger.error(message, exc_info=True)
        else:
            cls._logger.error(message)
    
    @classmethod
    def info(cls, message: str) -> None:
        """Print an info message in blue."""
        if not cls._tui_mode:
            cls._console.print(f"[blue]ℹ {message}[/blue]")
        cls._logger.info(message)
    
    @classmethod
    def progress(cls, message: str) -> None:
        """Print a progress message in cyan."""
        if not cls._tui_mode:
            cls._console.print(f"[cyan]⟳ {message}[/cyan]")
        cls._logger.info(f"PROGRESS: {message}")
    
    @classmethod
    def debug(cls, message: str) -> None:
        """Print a debug message in dim style."""
        if cls.debug_mode and not cls._tui_mode:
            cls._console.print(f"[dim]🔍 {message}[/dim]")
        cls._logger.debug(message)
    
    @classmethod
    def critical(cls, message: str) -> None:
        """Print a critical message in bold red."""
        if not cls._tui_mode:
            cls._console.print(f"[bold red]🚨 {message}[/bold red]")
        cls._logger.critical(message)
    
    @classmethod
    def print(cls, message: str, style: Optional[str] = None) -> None:
        """Print a message with optional style."""
        if not cls._tui_mode:
            if style:
                cls._console.print(f"[{style}]{message}[/{style}]")
            else:
                cls._console.print(message)
        cls._logger.info(f"PRINT: {message}")
    
    @classmethod
    def set_debug_mode(cls, enabled: bool) -> None:
        """Enable or disable debug mode."""
        cls.debug_mode = enabled
        if enabled:
            cls._logger.setLevel(logging.DEBUG)
            cls.info("Debug mode enabled")
        else:
            cls._logger.setLevel(logging.INFO)
            cls.info("Debug mode disabled")