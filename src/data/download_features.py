"""
Download pre-extracted features from HuggingFace: medarc/algonauts_2025.features

Usage:
    # Download all features
    python src/data/download_features.py

    # Download specific feature types only
    python src/data/download_features.py --features vjepa2_avg_feat whisper qwen-2-5-omni-7b

    # Download to a custom directory
    python src/data/download_features.py --data-dir /path/to/data

    # List available features without downloading
    python src/data/download_features.py --list
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, HfApi


REPO_ID = "medarc/algonauts_2025.features"
REPO_TYPE = "dataset"


def list_available_features():
    """List all available feature directories in the dataset."""
    api = HfApi()
    tree = api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, recursive=False)
    
    dirs = []
    for item in tree:
        if item.path != ".gitattributes":
            dirs.append(item.path)
    return sorted(dirs)


def download_features(data_dir: str, features: list[str] | None = None):
    """
    Download features from HuggingFace dataset.
    
    Args:
        data_dir: Target directory to save the downloaded features.
        features: List of specific feature names to download.
                  If None, downloads everything.
    """
    data_path = Path(data_dir).resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset: {REPO_ID}")
    print(f"Target directory: {data_path}")
    
    kwargs = dict(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(data_path),
        local_dir_use_symlinks=False,
    )
    
    if features:
        # Build allow_patterns to only download selected feature dirs
        allow_patterns = []
        for feat in features:
            allow_patterns.append(f"{feat}/**")
        kwargs["allow_patterns"] = allow_patterns
        print(f"Downloading features: {features}")
    else:
        print("Downloading ALL features...")
    
    print()
    path = snapshot_download(**kwargs)
    print(f"\nDownload complete! Files saved to: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-extracted features from HuggingFace (medarc/algonauts_2025.features)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Target directory to save features. Default: Data/algonauts_2025.features/",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        type=str,
        default=None,
        help="Specific feature type(s) to download (e.g., vjepa2_avg_feat whisper). "
             "If not specified, downloads everything.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available features and exit.",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print(f"Available features in {REPO_ID}:")
        for feat in list_available_features():
            print(f"  - {feat}")
        return
    
    # Default data dir
    if args.data_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        data_dir = project_root / "Data" / "algonauts_2025.features"
    else:
        data_dir = Path(args.data_dir)
    
    # Validate feature names if specified
    if args.features:
        available = list_available_features()
        for feat in args.features:
            if feat not in available:
                print(f"Error: '{feat}' is not a valid feature name.")
                print(f"Available features: {available}")
                sys.exit(1)
    
    download_features(str(data_dir), args.features)


if __name__ == "__main__":
    main()
