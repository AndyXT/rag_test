"""Decorators for the RAG CLI application"""

import functools
import time
import asyncio
import logging
from typing import Any, Callable, Optional, TypeVar


# Type variables for generic decorators
F = TypeVar("F", bound=Callable[..., Any])


def log_errors(
    logger: Optional[logging.Logger] = None,
    message: str = "Error in {func_name}",
    reraise: bool = True,
    default_return: Any = None,
) -> Callable[[F], F]:
    """
    Decorator to automatically log errors from functions

    Args:
        logger: Logger to use (defaults to function's module logger)
        message: Error message template
        reraise: Whether to re-raise the exception
        default_return: Value to return if exception is not re-raised

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get logger
                log = logger or logging.getLogger(func.__module__)

                # Format message
                error_msg = message.format(
                    func_name=func.__name__, error=str(e), error_type=type(e).__name__
                )

                # Log error
                log.error(error_msg, exc_info=True)

                if reraise:
                    raise
                return default_return

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Get logger
                log = logger or logging.getLogger(func.__module__)

                # Format message
                error_msg = message.format(
                    func_name=func.__name__, error=str(e), error_type=type(e).__name__
                )

                # Log error
                log.error(error_msg, exc_info=True)

                if reraise:
                    raise
                return default_return

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def log_performance(
    logger: Optional[logging.Logger] = None,
    message: str = "{func_name} completed in {duration:.2f}s",
    level: str = "DEBUG",
) -> Callable[[F], F]:
    """
    Decorator to log function performance

    Args:
        logger: Logger to use
        message: Message template
        level: Log level to use

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Get logger and level
                log = logger or logging.getLogger(func.__module__)
                log_level = getattr(logging, level.upper(), logging.DEBUG)

                # Log performance
                log.log(
                    log_level,
                    message.format(func_name=func.__name__, duration=duration),
                )

                return result
            except Exception:
                duration = time.time() - start_time
                log = logger or logging.getLogger(func.__module__)
                log.error(f"{func.__name__} failed after {duration:.2f}s")
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # Get logger and level
                log = logger or logging.getLogger(func.__module__)
                log_level = getattr(logging, level.upper(), logging.DEBUG)

                # Log performance
                log.log(
                    log_level,
                    message.format(func_name=func.__name__, duration=duration),
                )

                return result
            except Exception:
                duration = time.time() - start_time
                log = logger or logging.getLogger(func.__module__)
                log.error(f"{func.__name__} failed after {duration:.2f}s")
                raise

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None,
) -> Callable[[F], F]:
    """
    Decorator to retry function on failure

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch
        logger: Logger to use

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        log.error(
                            f"{func.__name__} failed after {max_attempts} attempts"
                        )
                        raise

                    log.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {str(e)}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            log = logger or logging.getLogger(func.__module__)
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        log.error(
                            f"{func.__name__} failed after {max_attempts} attempts"
                        )
                        raise

                    log.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {str(e)}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def validate_inputs(**validators):
    """
    Decorator to validate function inputs

    Args:
        **validators: Keyword arguments mapping parameter names to validator functions

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Validate inputs
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for {param_name}: {value}")

            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get function signature
            import inspect

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Validate inputs
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for {param_name}: {value}")

            return await func(*args, **kwargs)

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def deprecated(
    message: str = "This function is deprecated", version: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator to mark functions as deprecated

    Args:
        message: Deprecation message
        version: Version when deprecated

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import warnings

            warning_msg = f"{func.__name__}: {message}"
            if version:
                warning_msg += f" (deprecated since version {version})"

            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            import warnings

            warning_msg = f"{func.__name__}: {message}"
            if version:
                warning_msg += f" (deprecated since version {version})"

            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)
            return await func(*args, **kwargs)

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
