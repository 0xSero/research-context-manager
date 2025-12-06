"""Command-line interface for Cortex."""

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from cortex.config import CortexConfig, ModelConfig
from cortex.router import CortexRouter


console = Console()


async def run_interactive(config: CortexConfig) -> None:
    """Run Cortex in interactive mode."""
    console.print(Panel.fit(
        "[bold blue]Cortex[/bold blue] - Auxiliary Context Manager\n"
        f"Model A: {config.model_a.provider}/{config.model_a.model}\n"
        f"Model B: {config.model_b.provider}/{config.model_b.model}",
        title="Welcome",
    ))

    router = CortexRouter(config)

    try:
        console.print("[dim]Initializing...[/dim]")
        await router.initialize()
        console.print("[green]Ready![/green]\n")

        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")

                if user_input.lower() in ("exit", "quit", "q"):
                    break

                if user_input.lower() == "/stats":
                    stats = router.get_stats()
                    console.print(Panel(str(stats), title="Stats"))
                    continue

                if user_input.lower() == "/compact":
                    console.print("[dim]Compacting conversation...[/dim]")
                    summary = await router.compact_conversation()
                    console.print(Panel(Markdown(summary), title="Summary"))
                    continue

                if user_input.lower() == "/help":
                    console.print(Panel(
                        "/stats - Show statistics\n"
                        "/compact - Compact conversation\n"
                        "/help - Show this help\n"
                        "exit - Quit",
                        title="Commands",
                    ))
                    continue

                # Process the message
                console.print("\n[bold green]Assistant[/bold green]")
                async for chunk in router.process_message(user_input):
                    console.print(chunk, end="")
                console.print()  # Newline after response

            except KeyboardInterrupt:
                console.print("\n[dim]Use 'exit' to quit[/dim]")
                continue

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise

    finally:
        console.print("[dim]Shutting down...[/dim]")
        await router.shutdown()
        console.print("[green]Goodbye![/green]")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Cortex - Auxiliary Context Manager")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model for context management (Model A)",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model for work (Model B)",
    )
    parser.add_argument(
        "--provider-a",
        type=str,
        default="anthropic",
        help="Provider for Model A",
    )
    parser.add_argument(
        "--provider-b",
        type=str,
        default="anthropic",
        help="Provider for Model B",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Path to system prompt file",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        help="Path to MCP configuration file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    args = parser.parse_args()

    # Build configuration
    if args.config:
        # Load from file
        import json
        with open(args.config) as f:
            config_dict = json.load(f)
        config = CortexConfig(**config_dict)
    else:
        # Build from arguments
        config = CortexConfig(
            model_a=ModelConfig(
                provider=args.provider_a,
                model=args.model_a,
                temperature=0.3,
            ),
            model_b=ModelConfig(
                provider=args.provider_b,
                model=args.model_b,
                temperature=0.7,
            ),
            system_prompt_path=args.system_prompt,
            mcp_config_path=args.mcp_config,
            debug=args.debug,
        )

    # Run
    try:
        asyncio.run(run_interactive(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
