"""
Rich CLI Animations, Status Spinners, and Progress Displays.
"""

import asyncio
import itertools
import sys
import time
from typing import Callable, Optional
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

console = Console()


class LoadingAnimation:
    """Displays an informative loading animation with elapsed time and status updates."""

    @staticmethod
    async def run_with_spinner(
        message: str,
        coroutine_func: Callable,
        timeout_seconds: int = 30,
        spinner_name: str = "dots"
    ):
        start_time = time.time()
        spinner = Spinner(spinner_name, text=f"[cyan]{message}[/cyan]")

        with Live(spinner, refresh_per_second=10, transient=True) as live:
            # Run coroutine with timeout
            try:
                task = asyncio.create_task(coroutine_func())
                while not task.done():
                    elapsed = time.time() - start_time
                    spinner.update(text=f"[cyan]{message}[/cyan] [dim]({elapsed:.1f}s / max {timeout_seconds}s)[/dim]")
                    await asyncio.sleep(0.1)

                return await task
            except asyncio.TimeoutError:
                raise
