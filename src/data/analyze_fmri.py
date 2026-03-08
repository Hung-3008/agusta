import os
import h5py
from pathlib import Path

def analyze_h5_file(filepath):
    print(f"\n[{filepath.name}]")
    try:
        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            print(f"  Total keys (clips/runs): {len(keys)}")
            if len(keys) > 0:
                print("  Sample keys:", keys[:5])
                
                # Check the first key's shape
                first_key = keys[0]
                data = f[first_key]
                print(f"  Shape of '{first_key}': {data.shape}")
                print(f"  Data type: {data.dtype}")
                
            else:
                print("  File is empty or contains no keys.")
    except Exception as e:
        print(f"  Cannot open file. Error: {e}")
        print("  (Note: If this is a DataLad repository, make sure you ran 'datalad get' on the file.)")

def main():
    fmri_dir = Path("Data/algonauts_2025.competitors/fmri")
    
    if not fmri_dir.exists():
        print(f"Directory {fmri_dir} does not exist.")
        return
        
    print(f"Analyzing fMRI directory: {fmri_dir}")
    
    subjects = sorted([d.name for d in fmri_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    for subj in subjects:
        print(f"\n{'='*50}\nAnalyzing Subject: {subj}\n{'='*50}")
        func_dir = fmri_dir / subj / "func"
        
        if not func_dir.exists():
            print(f"  No 'func' directory found for {subj}.")
            continue
            
        h5_files = list(func_dir.glob("*.h5"))
        print(f"  Found {len(h5_files)} .h5 files.")
        
        for h5 in h5_files:
            analyze_h5_file(h5)

if __name__ == "__main__":
    main()
