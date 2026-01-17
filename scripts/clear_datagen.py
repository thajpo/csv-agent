#!/usr/bin/env python3
"""
Clear all generated datagen artifacts.

Removes:
- data/questions_synthetic/  (synthetic questions)
- data/questions/            (LLM questions)
- data/episodes/             (training episodes)
- data/datagen_manifest.jsonl (processing cache)

Usage:
    uv run python scripts/clear_datagen.py
    uv run python scripts/clear_datagen.py --dry-run  # preview what would be deleted
"""

import argparse
import shutil
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"

TARGETS = [
    DATA_DIR / "questions_synthetic",
    DATA_DIR / "questions",
    DATA_DIR / "episodes",
    DATA_DIR / "datagen_manifest.jsonl",
]


def get_size(path: Path) -> int:
    """Get total size of path in bytes."""
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return 0


def format_size(num_bytes: int) -> str:
    """Format bytes as human-readable."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Clear all generated datagen artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )
    args = parser.parse_args()

    total_size = 0
    found = []

    for target in TARGETS:
        if target.exists():
            size = get_size(target)
            total_size += size
            found.append((target, size))

    if not found:
        print("✓ Nothing to clear - all targets already empty")
        return 0

    print("Targets to clear:")
    for target, size in found:
        marker = "[dir]" if target.is_dir() else "[file]"
        print(f"  {marker} {target.relative_to(DATA_DIR.parent)} ({format_size(size)})")

    print(f"\nTotal: {format_size(total_size)}")

    if args.dry_run:
        print("\n[dry-run] No files deleted")
        return 0

    # Actually delete
    for target, _ in found:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        print(f"  ✓ Removed {target.name}")

    print(f"\n✓ Cleared {len(found)} targets ({format_size(total_size)})")
    return 0


if __name__ == "__main__":
    exit(main())
