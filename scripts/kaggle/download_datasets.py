#!/usr/bin/env python3
"""
Download CSV datasets from Kaggle with metadata.

Downloads to data/kaggle/{slug}/ with standardized structure:
    data/kaggle/{slug}/
    ├── data.csv
    └── meta.json

Usage:
    # Download from curated list (recommended)
    uv run python scripts/kaggle/download_datasets.py --from-list scripts/kaggle/curated_datasets.json

    # With size limit (default 5MB)
    uv run python scripts/kaggle/download_datasets.py --from-list scripts/kaggle/curated_datasets.json --max-size 10

    # Limit number of datasets
    uv run python scripts/kaggle/download_datasets.py --from-list scripts/kaggle/curated_datasets.json --limit 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Workaround for bug in kaggle 1.8.2 - it tries to delete KAGGLE_API_TOKEN
# even when reading from file (where the env var doesn't exist)
_orig_delitem = os.environ.__class__.__delitem__
os.environ.__class__.__delitem__ = lambda self, key: _orig_delitem(self, key) if key in self else None

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Error: kaggle package not installed.")
    print("Install with: uv sync --extra kaggle")
    sys.exit(1)


def get_api():
    """Return authenticated Kaggle API client.

    Supports KAGGLE_API_TOKEN env var (KGAT_* format).
    """
    api = KaggleApi()
    api.authenticate()
    return api


def get_dataset_metadata(api, dataset_ref: str) -> dict:
    """
    Fetch metadata for a dataset.

    Args:
        api: Authenticated Kaggle API instance
        dataset_ref: Dataset reference in format "owner/dataset-name"

    Returns:
        Dict with title, description, subtitle, tags, url, etc.
    """
    owner, dataset_name = dataset_ref.split("/")

    try:
        # Search for the dataset to get its metadata
        datasets = api.dataset_list(search=dataset_ref)
        dataset_info = None
        for d in datasets:
            if d.ref == dataset_ref:
                dataset_info = d
                break

        if dataset_info:
            # Extract tag names from tag objects
            tags = getattr(dataset_info, "tags", [])
            keywords = [t.get("name", t) if isinstance(t, dict) else str(t) for t in tags]

            return {
                "ref": dataset_ref,
                "owner": owner,
                "name": dataset_name,
                "title": getattr(dataset_info, "title", dataset_name),
                "subtitle": getattr(dataset_info, "subtitle", ""),
                "description": getattr(dataset_info, "description", ""),
                "keywords": keywords,
                "url": getattr(dataset_info, "url", f"https://www.kaggle.com/datasets/{dataset_ref}"),
                "totalBytes": getattr(dataset_info, "total_bytes", None),
                "downloadCount": getattr(dataset_info, "download_count", None),
            }
    except Exception as e:
        print(f"  Warning: Could not fetch metadata for {dataset_ref}: {e}")

    # Fallback with minimal info
    return {
        "ref": dataset_ref,
        "owner": owner,
        "name": dataset_name,
        "title": dataset_name,
        "subtitle": "",
        "description": "",
        "keywords": [],
        "url": f"https://www.kaggle.com/datasets/{dataset_ref}",
    }


def download_dataset(
    api,
    dataset_ref: str,
    output_dir: Path,
    max_size_mb: float = 5.0,
) -> Path | None:
    """
    Download dataset to output_dir/{slug}/ with standardized structure.

    Filters:
    - Skips multi-CSV datasets
    - Skips datasets larger than max_size_mb

    Returns:
        Path to dataset folder if successful, None if skipped.
    """
    import shutil

    slug = dataset_ref.replace("/", "_")
    temp_dir = output_dir / "_temp"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Download and unzip to temp
        api.dataset_download_files(dataset_ref, path=str(temp_dir), unzip=True)

        # Find CSV files
        csv_files = list(temp_dir.glob("**/*.csv"))

        if not csv_files:
            print(f"  ⚠ No CSV files found, skipping")
            return None

        # Filter: single-CSV only
        if len(csv_files) > 1:
            print(f"  ⚠ Multi-CSV dataset ({len(csv_files)} files), skipping")
            return None

        csv_file = csv_files[0]
        size_mb = csv_file.stat().st_size / (1024 * 1024)

        # Filter: file size
        if size_mb > max_size_mb:
            print(f"  ⚠ Too large ({size_mb:.1f} MB > {max_size_mb} MB), skipping")
            return None

        # Create dataset folder with standardized structure
        dataset_dir = output_dir / slug
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Move CSV as data.csv
        dest_csv = dataset_dir / "data.csv"
        shutil.move(str(csv_file), str(dest_csv))

        print(f"  ✓ Saved: {slug}/data.csv ({size_mb:.1f} MB)")
        return dataset_dir

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def get_popular_datasets(api) -> list[str]:
    """
    Get popular tabular CSV datasets from Kaggle.

    Returns list of dataset refs like "owner/dataset-name", sorted by votes.
    Fetches all available - caller handles limiting based on successful downloads.
    """
    try:
        datasets = api.dataset_list(
            file_type="csv",
            sort_by="votes",
            max_size=50 * 1024 * 1024,  # 50MB max (pre-filter)
        )

        return [d.ref for d in datasets]

    except Exception as e:
        print(f"Error fetching datasets: {e}")
        return []


def load_curated_list(path: Path) -> list[str]:
    """Load dataset refs from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    
    # Support both flat list and object with "datasets" key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "datasets" in data:
        return data["datasets"]
    else:
        raise ValueError(f"Invalid format in {path}. Expected list or {{datasets: [...]}}.")


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle datasets with metadata")
    parser.add_argument(
        "--from-list",
        type=Path,
        help="JSON file with curated dataset refs (default: fetch liked datasets)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of datasets to download"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent.parent / "data" / "kaggle",
        help="Output directory (default: data/kaggle/)"
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=5.0,
        help="Maximum CSV size in MB (default: 5.0)"
    )
    args = parser.parse_args()

    # Setup
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    api = get_api()
    print("✓ Kaggle API ready")
    print(f"  Output: {output_dir}")
    print(f"  Max size: {args.max_size} MB")
    print()

    # Get dataset list
    if args.from_list:
        print(f"Loading curated list from {args.from_list}...")
        dataset_refs = load_curated_list(args.from_list)
    else:
        print("Fetching popular tabular datasets...")
        dataset_refs = get_popular_datasets(api)

    print(f"Found {len(dataset_refs)} candidate datasets\n")

    # Download until we reach limit (counting only successful downloads)
    manifest = []
    skipped_filter = 0
    skipped_exists = 0
    successful_downloads = 0

    for i, ref in enumerate(dataset_refs, 1):
        # Stop when we have enough successful downloads
        if args.limit and successful_downloads >= args.limit:
            break

        print(f"[{i}] {ref} (got {successful_downloads}/{args.limit or '∞'})")

        # Skip if already downloaded
        slug = ref.replace("/", "_")
        existing_dir = output_dir / slug
        if (existing_dir / "data.csv").exists():
            print(f"  ✓ Already exists, counting as success")
            manifest.append({"ref": ref, "slug": slug})
            skipped_exists += 1
            successful_downloads += 1
            continue

        # Download (filters applied inside)
        dataset_dir = download_dataset(api, ref, output_dir, max_size_mb=args.max_size)

        if not dataset_dir:
            skipped_filter += 1
            continue

        # Get metadata and save inside dataset folder
        metadata = get_dataset_metadata(api, ref)
        meta_path = dataset_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        manifest.append({"ref": ref, "slug": slug})
        successful_downloads += 1

    # Save manifest at root level
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    newly_downloaded = successful_downloads - skipped_exists
    print(f"\n{'=' * 40}")
    print(f"✓ Successful: {successful_downloads} datasets")
    print(f"  New downloads: {newly_downloaded}")
    print(f"  Already existed: {skipped_exists}")
    print(f"  Filtered out: {skipped_filter}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
