from __future__ import annotations

import sys
from typing import Any

from pydantic import ValidationError


def pretty_print_error(e: ValidationError, title: str = "Configuration Error") -> None:
    """
    Format and display a Pydantic ValidationError in a beautiful,
    self-explanatory way using 'rich'.

    Falls back to a plain-text representation if 'rich' is not available.

    Args:
        e (ValidationError): The exception to display.
        title (str, optional): Title to show in the error panel. Defaults to "Configuration Error".
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console(stderr=True)
        error_count = e.error_count()
        s = "s" if error_count > 1 else ""

        # Build a table of field → error message
        table = Table(box=None, show_header=True, header_style="bold magenta")
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Error", style="red")

        for err in e.errors():
            # Join loc tuple with dots, e.g. ('optimizer', 'lr') -> 'optimizer.lr'
            loc = ".".join(str(p) for p in err["loc"])
            msg = err["msg"]
            table.add_row(loc, msg)

        # Wrap it in a panel with headers/subtitles
        panel = Panel(
            table,
            title=f"[bold red]{title}[/bold red]",
            subtitle=f"[bold red]{error_count} error{s} detected[/bold red]",
            border_style="red",
            expand=False,
        )
        console.print(panel)

    except ImportError:
        # Fallback if rich is missing
        print(f"{title}:", file=sys.stderr)
        print(f"{e.error_count()} errors detected:", file=sys.stderr)
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            msg = err["msg"]
            print(f"  - {loc}: {msg}", file=sys.stderr)
