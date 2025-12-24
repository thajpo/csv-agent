#!/usr/bin/env python3
"""
Download CSV datasets from Kaggle with metadata.

Usage:
    # Download user's liked datasets (default)
    uv run python kaggle/download_datasets.py

    # Download from a curated list
    uv run python kaggle/download_datasets.py --from-list kaggle/curated_datasets.json

    # Limit number of datasets
    uv run python kaggle/download_datasets.py --limit 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from kaggle import api as kaggle_api
except ImportError:
    print("Error: kaggle package not installed.")
    print("Install with: uv sync --extra kaggle")
    sys.exit(1)


def get_api():
    """Return authenticated Kaggle API client.

    Uses the new-style import which supports KAGGLE_API_TOKEN env var.
    """
    return kaggle_api


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


def download_dataset(api, dataset_ref: str, output_dir: Path) -> list[Path]:
    """
    Download dataset files to output directory.
    
    Returns list of downloaded CSV file paths.
    """
    # Create temp directory for this dataset
    temp_dir = output_dir / "_temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Download and unzip
        api.dataset_download_files(dataset_ref, path=str(temp_dir), unzip=True)
        
        # Find CSV files
        csv_files = list(temp_dir.glob("**/*.csv"))
        
        if not csv_files:
            print(f"  No CSV files found in {dataset_ref}")
            return []
        
        # Move CSV files to output directory with slug prefix
        slug = dataset_ref.replace("/", "_")
        downloaded = []
        
        for i, csv_file in enumerate(csv_files):
            if len(csv_files) == 1:
                dest_name = f"{slug}.csv"
            else:
                # Multiple CSVs: add suffix
                dest_name = f"{slug}_{csv_file.stem}.csv"
            
            dest_path = output_dir / dest_name
            csv_file.rename(dest_path)
            downloaded.append(dest_path)
            print(f"  Saved: {dest_name}")
        
        return downloaded
        
    finally:
        # Cleanup temp directory
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def get_liked_datasets(api, limit: int | None = None) -> list[str]:
    """
    Get user's liked/favorited datasets that are tabular.
    
    Returns list of dataset refs like "owner/dataset-name"
    """
    # Get user's datasets (this gets datasets they've interacted with)
    # Note: Kaggle API doesn't have a direct "liked" endpoint,
    # so we'll use dataset_list with user filter
    
    # For now, let's get popular tabular datasets as a fallback
    # The user can also provide a curated list
    
    try:
        # Try to get user's datasets first
        datasets = api.dataset_list(
            file_type="csv",
            sort_by="votes",
            max_size=50 * 1024 * 1024,  # 50MB max
        )
        
        refs = []
        for d in datasets:
            if limit and len(refs) >= limit:
                break
            refs.append(d.ref)
        
        return refs
        
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
        default=Path(__file__).parent / "downloaded",
        help="Output directory (default: kaggle/downloaded)"
    )
    args = parser.parse_args()
    
    # Setup
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    api = get_api()
    print("✓ Kaggle API ready\n")
    
    # Get dataset list
    if args.from_list:
        print(f"Loading curated list from {args.from_list}...")
        dataset_refs = load_curated_list(args.from_list)
    else:
        print("Fetching popular tabular datasets...")
        dataset_refs = get_liked_datasets(api, limit=args.limit)
    
    if args.limit:
        dataset_refs = dataset_refs[:args.limit]
    
    print(f"Found {len(dataset_refs)} datasets to download\n")
    
    # Download each dataset
    manifest = []
    
    for i, ref in enumerate(dataset_refs, 1):
        print(f"[{i}/{len(dataset_refs)}] {ref}")
        
        # Get metadata
        metadata = get_dataset_metadata(api, ref)
        
        # Download CSVs
        csv_files = download_dataset(api, ref, output_dir)
        
        if not csv_files:
            continue
        
        # Save metadata for each CSV
        slug = ref.replace("/", "_")
        meta_path = output_dir / f"{slug}.meta.json"
        
        metadata["csv_files"] = [f.name for f in csv_files]
        
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        manifest.append({
            "ref": ref,
            "slug": slug,
            "csv_files": [f.name for f in csv_files],
            "meta_file": meta_path.name,
        })
        
        print()
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Downloaded {len(manifest)} datasets")
    print(f"✓ Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
