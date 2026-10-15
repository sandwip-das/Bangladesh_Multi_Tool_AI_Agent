"""Interactive command-line interface."""
from __future__ import annotations

import argparse
from dotenv import load_dotenv

from src.agent import ask_agent
from src.db_tools import get_database_schema
from src.config import DATASETS


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Bangladesh Multi-Tool AI Agent")
    parser.add_argument("--schema", choices=["institutions", "hospitals", "restaurants"])
    parser.add_argument("--question", "-q", help="Ask one question and exit")
    args = parser.parse_args()

    if args.schema:
        print(get_database_schema(args.schema))
        return
    if args.question:
        print(ask_agent(args.question))
        return

    print("Bangladesh Multi-Tool AI Agent")
    print("Type a question, or type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue
        try:
            print("\nAgent:", ask_agent(question))
        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
