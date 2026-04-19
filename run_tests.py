#!/usr/bin/env python
"""AI Model Testing Framework - CLI entry point.

Usage:
  python run_tests.py [--model MODEL] [--fixture FIXTURE] [--list]

Exit codes: 0 = all pass, 1 = any failure or error.
"""
from __future__ import annotations
import argparse
import sys

from ai_test_framework.core.model import Model
from ai_test_framework.core.runner import TestSuite
from ai_test_framework.fixtures import FIXTURES

DEFAULT_MODEL = "llama3.2:1b"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI Model Testing Framework")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    p.add_argument("--fixture", help="Run a single fixture by name")
    p.add_argument("--list", action="store_true", help="List available fixtures and exit")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if args.list:
        for name in sorted(FIXTURES):
            print(f"  {name}")
        return 0

    if args.fixture and args.fixture not in FIXTURES:
        print(f"Unknown fixture: {args.fixture!r}. Run with --list to see available fixtures.", file=sys.stderr)
        return 1

    model = Model.ollama(args.model)
    suite = TestSuite("consistency")

    targets = [args.fixture] if args.fixture else sorted(FIXTURES)
    for name in targets:
        suite.add(FIXTURES[name]())

    result = suite.run(model)
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
