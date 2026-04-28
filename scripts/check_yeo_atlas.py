"""Check Schaefer 1000Par7Net atlas parcellation against the hardcoded
network_counts in NetworkSubjectLayers.

Standard Schaefer 1000 7-network ordering (each hemisphere):
  Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default
Labels 1-500 = LH, Labels 501-1000 = RH.

The standard per-hemisphere counts from the Schaefer 2018 paper are:
  Vis=75, SomMot=74, DorsAttn=66, SalVentAttn=68, Limbic=38, Cont=61, Default=118
"""
import sys
import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("nibabel not installed, trying to read with gzip+numpy...")
    sys.exit(1)

atlas_path = "Data/algonauts_2025.competitors/fmri/sub-01/atlas/sub-01_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
img = nib.load(atlas_path)
data = img.get_fdata()

print(f"Atlas shape: {data.shape}")

all_labels = data[data > 0].astype(int)
unique_labels = np.sort(np.unique(all_labels))
print(f"Total unique parcel labels: {len(unique_labels)}")
print(f"Label range: {unique_labels.min()} - {unique_labels.max()}")

# --- Standard Schaefer 1000Par7Net per-hemisphere boundaries ---
# The label ordering follows the Schaefer naming convention:
# Labels are sorted alphabetically by network name WITHIN each hemisphere.
# Network order in label numbering:
#   Cont, Default, DorsAttn, Limbic, SalVentAttn, SomMot, Vis
# (alphabetical)
# BUT the code uses the functional order:
#   Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default

# Standard Schaefer 2018 per-hemisphere counts (alphabetical label order):
SCHAEFER_ALPHA_ORDER = {
    'Cont': 61,
    'Default': 118,
    'DorsAttn': 66, 
    'Limbic': 38,
    'SalVentAttn': 68,
    'SomMot': 74,
    'Vis': 75,
}

# Code's order:
CODE_ORDER = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
CODE_PER_HEMI = [75, 74, 66, 68, 38, 61, 118]

# Let's verify from the atlas by looking at the label structure
# In Schaefer atlas, labels 1-500 are LH, 501-1000 are RH
lh_labels = unique_labels[unique_labels <= 500]
rh_labels = unique_labels[unique_labels > 500]
print(f"\nLH parcels: {len(lh_labels)} (labels {lh_labels.min()}-{lh_labels.max()})")
print(f"RH parcels: {len(rh_labels)} (labels {rh_labels.min()}-{rh_labels.max()})")

# The network boundaries for Schaefer 1000Par7Net are standard.
# Let's compute cumulative boundaries to verify the code's assumption.
# 
# The CRITICAL question is: does the code's concatenation order match
# how the fMRI data voxels are ordered?

# Let's check if there's a mapping file or if the data uses the raw label ordering
func_dir = "Data/algonauts_2025.competitors/fmri/sub-01/func"
stats_dir = "Data/algonauts_2025.competitors/fmri/sub-01/stats"

import os
print(f"\nfunc dir contents: {os.listdir(func_dir)[:5]}")
print(f"stats dir contents: {os.listdir(stats_dir)[:5]}")

# Check target_sample_number
tsn_dir = "Data/algonauts_2025.competitors/fmri/sub-01/target_sample_number"
if os.path.exists(tsn_dir):
    print(f"target_sample_number contents: {os.listdir(tsn_dir)}")

# The key insight: the fMRI target data is extracted as 1000 voxels
# corresponding to the 1000 Schaefer parcels. The order is by label number (1-1000).
# 
# Schaefer 1000Par7Net label numbering (alphabetical by network within hemisphere):
# LH_Cont_1 ... LH_Cont_61   -> labels 1-61
# LH_Default_1 ... LH_Default_118 -> labels 62-179
# LH_DorsAttn_1 ... LH_DorsAttn_66 -> labels 180-245
# LH_Limbic_1 ... LH_Limbic_38 -> labels 246-283
# LH_SalVentAttn_1 ... LH_SalVentAttn_68 -> labels 284-351
# LH_SomMot_1 ... LH_SomMot_74 -> labels 352-425
# LH_Vis_1 ... LH_Vis_75 -> labels 426-500
# RH_Cont_1 ... RH_Cont_61 -> labels 501-561
# ... etc.

# Label order (alphabetical) vs Code order:
alpha_order_names = ['Cont', 'Default', 'DorsAttn', 'Limbic', 'SalVentAttn', 'SomMot', 'Vis']
alpha_order_counts = [61, 118, 66, 38, 68, 74, 75]  # per hemisphere

# Code order:
code_order_names  = ['Visual', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
code_order_counts = [75, 74, 66, 68, 38, 61, 118]  # per hemisphere

print("\n" + "="*70)
print("COMPARISON: Atlas label order vs Code order")
print("="*70)

print("\nSchaefer atlas LABEL ORDER (alphabetical, labels 1→500 then 501→1000):")
cum = 0
for name, cnt in zip(alpha_order_names, alpha_order_counts):
    print(f"  LH_{name:15s}: labels {cum+1:4d} - {cum+cnt:4d}  ({cnt} parcels)")
    cum += cnt
for name, cnt in zip(alpha_order_names, alpha_order_counts):
    print(f"  RH_{name:15s}: labels {cum+1:4d} - {cum+cnt:4d}  ({cnt} parcels)")
    cum += cnt

print(f"\nCode's NetworkSubjectLayers OUTPUT ORDER:")
cum = 0
for name, cnt in zip(code_order_names, [2*c for c in code_order_counts]):
    print(f"  {name:15s}: voxels {cum:4d} - {cum+cnt-1:4d}  ({cnt} parcels, both hemi)")
    cum += cnt

print(f"\n{'='*70}")
print("MISMATCH ANALYSIS:")
print(f"{'='*70}")

# If fMRI data uses label ordering (1-1000), then:
# Data voxel 0 = LH_Cont parcel 1
# Data voxel 60 = LH_Cont parcel 61  
# Data voxel 61 = LH_Default parcel 1
# ...etc

# But code assumes:
# Output voxel 0 = Visual network
# Output voxel 149 = last Visual voxel
# Output voxel 150 = SomMot network
# ...etc

# These are DIFFERENT orderings!
data_order = []
for name, cnt in zip(alpha_order_names, alpha_order_counts):
    data_order.extend([f"LH_{name}"] * cnt)
for name, cnt in zip(alpha_order_names, alpha_order_counts):
    data_order.extend([f"RH_{name}"] * cnt)

code_output_order = []
for name, cnt in zip(code_order_names, [2*c for c in code_order_counts]):
    code_output_order.extend([name] * cnt)

# Map code names to comparable names
name_map = {'Visual': 'Vis', 'SomMot': 'SomMot', 'DorsAttn': 'DorsAttn', 
            'SalVentAttn': 'SalVentAttn', 'Limbic': 'Limbic', 'Cont': 'Cont', 'Default': 'Default'}

print("\nFirst 10 voxels - Data vs Code output:")
for i in range(10):
    code_net = name_map.get(code_output_order[i], code_output_order[i])
    data_net = data_order[i].split('_', 1)[1]
    match = "✓" if code_net == data_net else "✗ MISMATCH"
    print(f"  Voxel {i:4d}: Data={data_order[i]:20s} | Code={code_output_order[i]:15s} {match}")

# Count mismatches
mismatches = 0
for i in range(1000):
    code_net = name_map.get(code_output_order[i], code_output_order[i])
    data_net = data_order[i].split('_', 1)[1]
    if code_net != data_net:
        mismatches += 1

print(f"\nTotal mismatches: {mismatches} / 1000 voxels")
if mismatches > 0:
    print("⚠️  WARNING: Code's network head ordering does NOT match atlas label ordering!")
    print("   The code concatenates heads as [Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default]")
    print("   But atlas labels are ordered alphabetically: [Cont, Default, DorsAttn, Limbic, SalVentAttn, SomMot, Vis]")
    print("   Additionally, the code merges both hemispheres per network,")
    print("   while the atlas has all LH labels first (1-500), then all RH labels (501-1000).")
else:
    print("✓ Code's network head ordering matches atlas label ordering.")
