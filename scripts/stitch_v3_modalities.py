import argparse
import sys
from pathlib import Path

import torch

# Add src to path to import models
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.models.brainflow.brain_flow_direct_v3 import BrainFlowDirectV3

def get_v3_config(num_modalities=8):
    # Dummy modality dims just to establish shapes
    return {
        "modality_dims": [2048] * num_modalities,
        "hidden_dim": 1024,
        "proj_dim": 256,
        "max_seq_len": 31,
    }

def main():
    parser = argparse.ArgumentParser(description="Stitch V3 weights for new modalities")
    parser.add_argument("--old_ckpt", type=str, required=True, help="Path to best.pt with 8 modalities")
    parser.add_argument("--output", type=str, required=True, help="Path to save stitched.pt")
    args = parser.parse_args()

    print(f"Loading old checkpoint: {args.old_ckpt}")
    old_ckpt = torch.load(args.old_ckpt, map_location="cpu")
    old_state = old_ckpt.get("model", old_ckpt)

    # Build a config dict matching the kwargs of BrainFlowDirectV3.__init__
    modality_dims = [1408, 1280, 2048, 3072, 2048, 3584, 3584, 5120, 3584]
    vn_params = {
        "output_dim": 1000,
        "hidden_dim": 1024,
        "proj_dim": 256,
        "modality_dims": modality_dims,
        "n_blocks": 4,
        "n_heads": 8,
        "dropout": 0.1,
        "modality_dropout": 0.3,
        "max_seq_len": 31,
        "temporal_attn_layers": 2
    }
    
    # We must instantiate it exactly as the training script does:
    new_model = BrainFlowDirectV3(
        output_dim=1000,
        velocity_net_params=vn_params,
        n_subjects=4,
        reg_weight=0.5
    )
    new_state = new_model.state_dict()

    stitched_state = {}
    copied_exact = 0
    stitched = 0
    skipped = 0

    print("\nStitching weights...")
    for key, new_tensor in new_state.items():
        if key in old_state:
            old_tensor = old_state[key]
            
            if old_tensor.shape == new_tensor.shape:
                stitched_state[key] = old_tensor.clone()
                copied_exact += 1
            else:
                # Shape mismatch! (e.g. fusion_block.output_proj.0.weight)
                print(f"  Shape mismatch for {key}: old={list(old_tensor.shape)} new={list(new_tensor.shape)}")
                
                # Check if it's the weight matrix we expect to expand on dim 1
                if old_tensor.dim() == 2 and new_tensor.dim() == 2:
                    if old_tensor.shape[0] == new_tensor.shape[0] and old_tensor.shape[1] < new_tensor.shape[1]:
                        old_feat_dim = old_tensor.shape[1]
                        
                        # Start with new model's random initialization
                        stitched_tensor = new_tensor.clone()
                        
                        # Overwrite the first `old_feat_dim` columns with the old trained weights
                        stitched_tensor[:, :old_feat_dim] = old_tensor
                        
                        # Scale down the new random weights so they don't overpower the trained ones initially
                        # A common trick is to initialize new connections near zero 
                        # so the model slowly learns to incorporate the new modality
                        stitched_tensor[:, old_feat_dim:] *= 0.01 
                        
                        stitched_state[key] = stitched_tensor
                        stitched += 1
                        print(f"    -> Stitched! Kept {old_feat_dim} old dims, appended {new_tensor.shape[1] - old_feat_dim} new dims (scaled 0.01x)")
                    else:
                        print("    -> Incompatible dimension expansion. Using random init.")
                        stitched_state[key] = new_tensor
                        skipped += 1
                else:
                    print("    -> Not a 2D matrix. Using random init.")
                    stitched_state[key] = new_tensor
                    skipped += 1
        else:
            print(f"  New key in model (Random Init): {key} {list(new_tensor.shape)}")
            stitched_state[key] = new_tensor
            skipped += 1

    print(f"\nDone. Copied={copied_exact}, Stitched={stitched}, Skipped/Random={skipped}")
    
    # Save stitched checkpoint
    # We only save the model state so the training script starts fresh optimizer/scheduler
    # from Epoch 1, but with perfectly warm-started model weights.
    torch.save({"model": stitched_state}, args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
