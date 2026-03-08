"""Quick script to inspect feature H5 file structure."""
import h5py
from pathlib import Path

def inspect_h5(path):
    print(f"\n[{Path(path).name}]")
    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            print(f"  Top-level keys (first 3): {keys[:3]}")
            if keys:
                # inspect first clip
                clip = f[keys[0]]
                if hasattr(clip, 'keys'):
                    sub_keys = list(clip.keys())
                    print(f"  Sub-keys under '{keys[0]}': {sub_keys}")
                    if sub_keys:
                        shape = clip[sub_keys[0]].shape
                        print(f"  Shape of '{keys[0]}/{sub_keys[0]}': {shape}")
                else:
                    print(f"  Shape of '{keys[0]}': {clip.shape}")
    except Exception as e:
        print(f"  Error: {e}")

# Check one file per modality
inspect_h5("Data/features/video/friends_s1_features_vjepa2.h5")
inspect_h5("Data/features/audio/friends_s1_features_w2vbert.h5")
inspect_h5("Data/features/text/friends_s1_features_llama3.h5")
inspect_h5("Data/features/omni/friends_s1_features_qwen_omni.h5")
