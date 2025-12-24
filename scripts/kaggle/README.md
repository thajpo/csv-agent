# Kaggle Dataset Downloader

Download CSV datasets from Kaggle with their metadata for training data generation.

## Setup

### 1. Install Kaggle API

The kaggle package is an optional dependency. Install it with:

```bash
uv sync --extra kaggle
```

### 2. Configure API Credentials

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings/account)
2. Scroll to "API" section and click **"Create New API Token"**
3. This downloads `kaggle.json` - move it to the right location:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Usage

### Download Popular Tabular Datasets (Default)

```bash
uv run python kaggle/download_datasets.py
```

This downloads popular CSV datasets from Kaggle.

### Download from a Curated List

Create a JSON file with dataset refs:

```json
{
  "datasets": [
    "hugomathien/soccer",
    "uciml/iris",
    "shivamb/netflix-shows"
  ]
}
```

Then run:

```bash
uv run python kaggle/download_datasets.py --from-list kaggle/curated_datasets.json
```

### Limit Number of Datasets

```bash
uv run python kaggle/download_datasets.py --limit 10
```

## Output Structure

Downloaded datasets appear in `kaggle/downloaded/`:

```
kaggle/downloaded/
├── hugomathien_soccer.csv
├── hugomathien_soccer.meta.json
├── uciml_iris.csv
├── uciml_iris.meta.json
└── manifest.json
```

### Metadata Format

Each `*.meta.json` contains:

```json
{
  "ref": "owner/dataset-name",
  "title": "Dataset Title",
  "subtitle": "Short description",
  "description": "Full dataset description...",
  "keywords": ["tag1", "tag2"],
  "url": "https://www.kaggle.com/datasets/owner/dataset-name",
  "csv_files": ["owner_dataset-name.csv"]
}
```

### Manifest

`manifest.json` lists all downloaded datasets for easy iteration:

```json
[
  {
    "ref": "owner/dataset-name",
    "slug": "owner_dataset-name",
    "csv_files": ["owner_dataset-name.csv"],
    "meta_file": "owner_dataset-name.meta.json"
  }
]
```

## Using Downloaded Datasets

Point your training pipeline at downloaded CSVs:

```bash
# Generate questions for a Kaggle dataset
uv run python scripts/generate_questions.py --csv kaggle/downloaded/hugomathien_soccer.csv
```

The metadata in `*.meta.json` provides context for question generation.
