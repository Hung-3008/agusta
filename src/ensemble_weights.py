import argparse
import glob
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ensemble_weights")

def load_state_dict(ckpt_path):
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Support both full checkpoint (with "model" key) and raw state_dict
    state = ckpt.get("model", ckpt)
    return state

def ensemble_checkpoints(checkpoint_paths, output_path):
    if not checkpoint_paths:
        logger.error("No checkpoints provided for ensembling.")
        return

    n_models = len(checkpoint_paths)
    logger.info(f"Ensembling {n_models} models using uniform weight averaging.")

    # Load the first model to initialize the ensembled state dict
    ensembled_state = load_state_dict(checkpoint_paths[0])
    
    # Initialize all tensors to float32 to avoid overflow during sum
    for k, v in ensembled_state.items():
        ensembled_state[k] = v.clone().float()

    # Add weights from the remaining models
    for path in checkpoint_paths[1:]:
        state = load_state_dict(path)
        for k, v in state.items():
            if k not in ensembled_state:
                logger.warning(f"Key {k} found in {path} but not in base checkpoint.")
                continue
            ensembled_state[k] += v.float()

    # Divide by N to get the average
    logger.info(f"Averaging weights (dividing by {n_models})...")
    for k, v in ensembled_state.items():
        ensembled_state[k] = v / n_models
        
        # Convert back to the original dtype of the first checkpoint if it was bfloat16 or float16
        # Or just keep it as is, but BrainFlow usually uses float32 for stored weights
        # Actually, PyTorch's default save is float32. We'll leave it as float32.

    # Save
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving ensembled checkpoint to: {out_file}")
    torch.save(ensembled_state, out_file)
    logger.info("Ensembling complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble PyTorch checkpoints via Weight Averaging (Model Soups)")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint paths to ensemble")
    parser.add_argument("--output", type=str, required=True, help="Output path for the ensembled checkpoint")
    args = parser.parse_args()

    # Expand any globs in the checkpoints list just in case the shell didn't
    expanded_paths = []
    for pattern in args.checkpoints:
        matches = glob.glob(pattern)
        if matches:
            expanded_paths.extend(matches)
        else:
            # If no glob match, add the raw string (maybe it's an exact path that doesn't exist yet, let load_state_dict fail)
            expanded_paths.append(pattern)
    
    # Remove duplicates but preserve order roughly
    unique_paths = []
    seen = set()
    for p in expanded_paths:
        abs_p = str(Path(p).resolve())
        if abs_p not in seen:
            seen.add(abs_p)
            unique_paths.append(p)

    ensemble_checkpoints(unique_paths, args.output)
