import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from agent.agent import SalesAgent

console = Console()

BANNER = """
╔══════════════════════════════════════════╗
║      Sales Call Copilot  🎯             ║
║  Ask anything about your sales calls    ║
╚══════════════════════════════════════════╝

Commands:
  exit / quit    → end session
  reset          → clear conversation history
  verbose        → toggle tool call visibility
  ingest <path>  → quick ingest shortcut

Examples:
  list my call ids
  summarise the last call
  what objections did prospects raise across all calls?
  give me all negative comments when pricing was mentioned
  which calls mentioned Competitor X?
"""


def run_cli():
    console.print(Panel(BANNER, style="bold cyan"))

    agent   = SalesAgent()
    verbose = False

    while True:
        try:
            # ── User input ────────────────────────────────────────────────────
            user_input = console.input("\n[bold green]You:[/bold green] ").strip()

            if not user_input:
                continue

            # ── Built-in commands ─────────────────────────────────────────────
            if user_input.lower() in ("exit", "quit"):
                console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]")
                sys.exit(0)

            if user_input.lower() == "reset":
                agent.reset()
                console.print("[yellow]Conversation history cleared.[/yellow]")
                continue

            if user_input.lower() == "verbose":
                verbose = not verbose
                state = "ON" if verbose else "OFF"
                console.print(f"[yellow]Verbose mode {state} — tool calls will {'be shown' if verbose else 'be hidden'}.[/yellow]")
                continue

            if user_input.lower().startswith("ingest "):
                path = user_input[7:].strip()
                user_input = f"ingest a new call transcript from {path}"

            # ── Agent call ────────────────────────────────────────────────────
            with console.status("[bold yellow]Thinking...[/bold yellow]", spinner="dots"):
                response = agent.chat(user_input, verbose=verbose)

            # ── Render response as markdown ───────────────────────────────────
            console.print("\n[bold blue]Copilot:[/bold blue]")
            console.print(Markdown(response))

        except KeyboardInterrupt:
            console.print("\n\n[bold cyan]Interrupted. Type 'exit' to quit.[/bold cyan]")
            continue

        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            continue


if __name__ == "__main__":
    run_cli()