#!/usr/bin/env python3
"""Convert H5 multimodal features to fast-loading NPY format.

Extracts all valid layers from H5 files and saves them as NPY arrays
with shape (T, n_layers, D). This allows fast loading with PyTorch
DataLoaders while preserving the ability to use WeightedLayerPool.

Usage:
    python src/data/convert_h5_to_npy_multilayer.py
"""

import argparse
import logging
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("convert_h5_to_npy_multilayer")

# Configuration for all 8 modalities
MODALITIES = {
    'video':   {'subdir': 'vjepa2_avg_feat',    'dim': 1408, 'expected_layers': 13},
    'audio':   {'subdir': 'whisper',            'dim': 1280, 'expected_layers': 4},
    'text_1b': {'subdir': 'Llama-3.2-1B',       'dim': 2048, 'expected_layers': 3},
    'text_3b': {'subdir': 'Llama-3.2-3B',       'dim': 3072, 'expected_layers': 5},
    'omni':    {'subdir': 'qwen-2-5-omni-7b',   'dim': 3584, 'expected_layers': 9},
    'vlm_14b': {'subdir': 'InternVL3_14B',      'dim': 5120, 'expected_layers': 5},
    'vlm_8b':  {'subdir': 'internvl3_8b_8bit',  'dim': 3584, 'expected_layers': 4},
    'text_qw': {'subdir': 'qwen2-5_3B',         'dim': 2048, 'expected_layers': 7},
}

def process_file(args):
    """Worker function to convert a single H5 file."""
    h5_path, npy_path, expected_dim, expected_layers = args
    
    if npy_path.exists():
        return True, "already exists"

    try:
        with h5py.File(h5_path, 'r') as f:
            layer_keys = sorted(f.keys())
            layers_data = []

            for lk in layer_keys:
                arr = f[lk][:].astype(np.float32)  # (T, [1,] D)
                
                # Squeeze singleton token dim if present
                if arr.ndim == 3 and arr.shape[1] == 1:
                    arr = arr.squeeze(1)  # (T, D)
                    
                # Skip invalid/degenerate layers
                try:
                    std_val = float(np.std(arr))
                    if not np.isfinite(std_val) or std_val == 0:
                        continue
                except (OverflowError, FloatingPointError):
                    continue
                    
                if arr.shape[1] != expected_dim:
                    return False, f"wrong dim {arr.shape[1]} != {expected_dim}"
                    
                layers_data.append(arr)

            if not layers_data:
                return False, "no valid layers found"

            # Stack: (L, T, D) → (T, L, D)
            stacked = np.stack(layers_data, axis=0)
            stacked = stacked.transpose(1, 0, 2)
            
            # Clean up NaNs/Infs
            stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Check shape matches expected layer count
            if stacked.shape[1] != expected_layers:
                return False, f"got {stacked.shape[1]} layers, expected {expected_layers} for {h5_path.name}"

            # Save
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, stacked)
            
        return True, "ok"
    
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="Data/algonauts_2025.features", help="Root dir of H5 features")
    parser.add_argument("--output_dir", default="Data/features_npy_multilayer", help="Output root for NPY")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input dir not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    
    # Scan all modalities and files
    for mod_name, cfg in MODALITIES.items():
        mod_in_dir = input_dir / cfg['subdir']
        mod_out_dir = output_dir / cfg['subdir']
        
        if not mod_in_dir.exists():
            logger.warning(f"Modality dir missing: {mod_in_dir}")
            continue
            
        logger.info(f"Scanning {mod_name}...")
        
        for root, _, files in os.walk(mod_in_dir):
            root_path = Path(root)
            # Maintain relative directory structure (e.g., friends/s1/...)
            rel_path = root_path.relative_to(mod_in_dir)
            out_path = mod_out_dir / rel_path
            
            for f in files:
                if f.endswith('.h5'):
                    h5_path = root_path / f
                    npy_path = out_path / f.replace('.h5', '.npy')
                    tasks.append((
                        h5_path, 
                        npy_path, 
                        cfg['dim'], 
                        cfg['expected_layers']
                    ))

    logger.info(f"Found {len(tasks)} files to convert.")
    
    if not tasks:
        return

    # Process in parallel
    # Use almost all available CPUs (leave 2 for OS)
    num_workers = max(4, (os.cpu_count() or 8) - 2)
    logger.info(f"Starting conversion with {num_workers} workers (chunksize=10)...")
    
    success = 0
    failed = 0
    skipped = 0
    
    with Pool(num_workers) as pool:
        for ok, msg in tqdm(pool.imap_unordered(process_file, tasks, chunksize=10), total=len(tasks)):
            if ok:
                if msg == "already exists":
                    skipped += 1
                else:
                    success += 1
            else:
                failed += 1
                # To debug failures, uncomment below
                # logger.warning(f"Failed: {msg}")

    logger.info(f"Conversion complete!")
    logger.info(f"  Success: {success}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Failed:  {failed}")
    logger.info(f"\nUpdate 'features_dir' in yaml to: {output_dir}")


if __name__ == "__main__":
    main()
