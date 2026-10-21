#!/usr/bin/env python3
"""Run the evaluation CLI with a small default sample count and explicit artifacts."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--num-test-videos", type=int, default=5)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=Path("test_evaluation"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for name in ("checkpoint", "config"):
        path = getattr(args, name)
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            parser.error(f"{name} not found: {path}")
    command = [sys.executable, str(root / "evaluate_flow_matching.py")]
    for key, value in vars(args).items():
        if value is not None:
            command.extend((f"--{key.replace('_', '-')}", str(value)))
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
