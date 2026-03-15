"""
Rich console helpers for the Trumpet Outreach Engine.
"""

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()


def header(title: str, subtitle: str = "") -> None:
    text = Text(title, style="bold white")
    if subtitle:
        text.append(f"\n{subtitle}", style="dim")
    console.print(Panel(text, style="bold blue", border_style="blue", padding=(0, 2)))


def section(title: str) -> None:
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))


def info(msg: str) -> None:
    console.print(f"  [dim]›[/dim] {msg}")


def success(msg: str) -> None:
    console.print(f"  [green]✓[/green] {msg}")


def warn(msg: str) -> None:
    console.print(f"  [yellow]⚠[/yellow]  {msg}")


def error(msg: str) -> None:
    console.print(f"  [red]✗[/red] {msg}")


def stream_thinking(text: str) -> None:
    console.print(text, end="", style="dim italic")


def stream_text(text: str) -> None:
    console.print(text, end="", style="white")


def print_leads_table(leads: list[dict]) -> None:
    table = Table(box=box.ROUNDED, border_style="cyan", header_style="bold cyan")
    table.add_column("Name", style="bold white")
    table.add_column("Title")
    table.add_column("Why Target")
    table.add_column("LinkedIn / Contact")
    for lead in leads:
        table.add_row(
            lead.get("name", ""),
            lead.get("title", ""),
            lead.get("why_target", ""),
            lead.get("linkedin_url", lead.get("contact_hint", "")),
        )
    console.print(table)


def print_intent_table(signals: list[dict]) -> None:
    table = Table(box=box.ROUNDED, border_style="yellow", header_style="bold yellow")
    table.add_column("Signal", style="bold white")
    table.add_column("Strength", justify="center")
    table.add_column("Evidence")
    for sig in signals:
        strength = sig.get("strength", "medium")
        color = {"high": "green", "medium": "yellow", "low": "dim"}.get(strength, "white")
        table.add_row(
            sig.get("signal", ""),
            f"[{color}]{strength.upper()}[/{color}]",
            sig.get("evidence", ""),
        )
    console.print(table)


def score_badge(score: int) -> str:
    if score >= 75:
        return f"[bold green]{score}/100 🔥 HOT[/bold green]"
    if score >= 50:
        return f"[bold yellow]{score}/100 ⚡ WARM[/bold yellow]"
    if score >= 25:
        return f"[yellow]{score}/100 LUKEWARM[/yellow]"
    return f"[dim]{score}/100 COLD[/dim]"
