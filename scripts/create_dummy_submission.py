"""
Create a dummy submission for the Algonauts 2025 Challenge on Codabench.
Generates random predictions in the correct format to test the submission pipeline.

Submission format (matching TRIDE's Benchmark callback):
- submission.npy: dict {subject -> {chunk -> np.array(n_samples, 1000)}}
- submission.zip: zipped version of submission.npy

Usage:
    python scripts/create_dummy_submission.py
"""

import numpy as np
import zipfile
from pathlib import Path


def create_dummy_submission(
    data_dir: str = "Data/algonauts_2025.competitors",
    output_dir: str = "results/dummy_submission",
):
    """Create a dummy submission with random predictions."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = ["sub-01", "sub-02", "sub-03", "sub-05"]
    n_parcels = 1000  # Schaefer 1000 parcels

    submission_dict = {}

    for subject in subjects:
        # Load target sample numbers for Season 7
        samples_file = (
            data_dir / "fmri" / subject / "target_sample_number"
            / f"{subject}_friends-s7_fmri_samples.npy"
        )
        target_sample_number = np.load(samples_file, allow_pickle=True).item()

        submission_dict[subject] = {}
        total_samples = 0

        for chunk, n_samples in sorted(target_sample_number.items()):
            # Create random predictions: shape (n_samples, n_parcels)
            predictions = np.random.randn(n_samples, n_parcels).astype(np.float32)
            submission_dict[subject][chunk] = predictions
            total_samples += n_samples

        print(f"{subject}: {len(target_sample_number)} chunks, {total_samples} total samples")

    # Save submission.npy
    submission_path = output_dir / "submission.npy"
    np.save(submission_path, submission_dict)
    print(f"\nSaved submission.npy: {submission_path}")
    print(f"File size: {submission_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Create submission.zip
    zip_path = submission_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(submission_path, arcname="submission.npy")
    print(f"Saved submission.zip: {zip_path}")
    print(f"Zip size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Verify the submission
    print("\n--- Verification ---")
    loaded = np.load(submission_path, allow_pickle=True).item()
    for subject in subjects:
        assert subject in loaded, f"Missing subject: {subject}"
        samples_file = (
            data_dir / "fmri" / subject / "target_sample_number"
            / f"{subject}_friends-s7_fmri_samples.npy"
        )
        target = np.load(samples_file, allow_pickle=True).item()
        for chunk, expected_n in target.items():
            assert chunk in loaded[subject], f"Missing chunk: {subject}/{chunk}"
            actual_shape = loaded[subject][chunk].shape
            assert actual_shape == (expected_n, n_parcels), \
                f"Shape mismatch for {subject}/{chunk}: {actual_shape} != ({expected_n}, {n_parcels})"
    print("All checks passed! Submission is valid.")
    print(f"\nUpload this file to Codabench: {zip_path}")


if __name__ == "__main__":
    create_dummy_submission()
