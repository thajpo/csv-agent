"""Batch runner for procedural (program-based) question generation."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.core.config import config
from src.datagen.synthetic.programs.program_generator import run_pipeline


async def run_all(max_datasets: int | None = None) -> int:
    csv_sources = config.csv_sources
    if isinstance(csv_sources, str):
        csv_sources = [csv_sources]

    if max_datasets is not None:
        csv_sources = csv_sources[:max_datasets]

    if not csv_sources:
        print("No CSV sources configured.")
        return 1

    generated = 0
    for csv_path in csv_sources:
        questions = await run_pipeline(csv_path=csv_path)
        generated += len(questions)

    print(f"Generated {generated} procedural questions across {len(csv_sources)} datasets")
    return 0 if generated > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run procedural question generation")
    parser.add_argument("--max-datasets", type=int, default=None)
    args = parser.parse_args()
    return asyncio.run(run_all(max_datasets=args.max_datasets))


if __name__ == "__main__":
    raise SystemExit(main())
