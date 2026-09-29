#!/usr/bin/env python3
"""
ReAct Agent CLI - Entry point for terminal interaction.
"""

from rich.console import Console
from rich.markdown import Markdown

from agent_core import run_prompt

console = Console()

if __name__ == "__main__":
    print("\n👋 Welcome to the ReAct demo! Type ‘exit’ or Ctrl‑C to quit.\n")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            answer = run_prompt(user_input)

            console.print("\n[bold blue]Agent:[/bold blue]")
            console.print(Markdown(answer))
            print()
        except KeyboardInterrupt:
            print("\nInterrupted – see you next time!")
            break
