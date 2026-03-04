import os
import subprocess
from pathlib import Path
import sys

def download_algonauts_data(data_dir):
    """
    Downloads the Algonauts 2025 Challenge dataset using DataLad.
    The dataset contains multimodal movies, audio, and fMRI responses.
    """
    data_path = Path(data_dir).resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Target data directory: {data_path}")
    
    # Check if datalad is installed
    try:
        subprocess.run(["datalad", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: DataLad is not installed or not found in PATH.")
        print("To download the challenge data, you first need to install DataLad.")
        print("See installation instructions: https://handbook.datalad.org/en/latest/intro/installation.html")
        print("For instance, using conda: conda install -c conda-forge datalad")
        sys.exit(1)
    
    repo_path = data_path / "algonauts_2025.competitors"
    
    # Install the dataset via datalad
    if not repo_path.exists():
        print(f"Installing DataLad dataset from Github to '{repo_path}'...")
        try:
            subprocess.run([
                "datalad", "install", "-r", "-s", 
                "https://github.com/courtois-neuromod/algonauts_2025.competitors.git",
                str(repo_path)
            ], check=True)
            print("Dataset repository installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dataset: {e}")
            sys.exit(1)
    else:
        print(f"Dataset repository already exists at '{repo_path}'.")

    # Download the actual data
    print("Downloading the challenge dataset (this will download multiple GBs of fMRI/movie data)...")
    print("Using 8 parallel jobs (-J8) to speed up the download.")
    try:
        subprocess.run([
            "datalad", "get", "-r", "-J8", "."
        ], cwd=str(repo_path), check=True)
        print("\nAll data successfully downloaded!")
    except subprocess.CalledProcessError as e:
        print(f"\nError occurred while downloading data: {e}")
        print("Note: You may retry the script to resume downloading.")

if __name__ == "__main__":
    # Determine the project root (assuming script is in src/data/)
    # and the target Data/ directory (which is at the project root)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    target_data_dir = project_root / "Data"
    
    download_algonauts_data(target_data_dir)
