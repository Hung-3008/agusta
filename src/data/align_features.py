import argparse
import os
import numpy as np
import h5py
import random
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def process_pca(args, seed, raw_base_dir, out_pca_dir):
    """
    Applies Z-scoring and PCA on train features and transforms test features.
    Saves 'features_train.npy' and 'features_test.npy' containing dicts of {movie_split: chunks}.
    """
    print(f"\n--- Running PCA downsampling for {args.modality} ---")
    
    # Training splits
    stimuli_list = []
    for i in range(1, 7):
        path = raw_base_dir / 'friends' / args.modality / f'friends_s{i}_features_{args.modality}.h5'
        if path.exists(): stimuli_list.append(path)
    for m in ['bourne', 'figures', 'life', 'wolf']:
        path = raw_base_dir / 'movie10' / args.modality / f'movie10_{m}_features_{args.modality}.h5'
        if path.exists(): stimuli_list.append(path)

    movie_splits = []
    chunks_per_movie = []
    features_list = []

    for stim_dir in stimuli_list:
        with h5py.File(stim_dir, 'r') as data:
            for movie in data.keys():
                if args.modality != 'language':
                    feat = np.asarray(data[movie][args.modality])
                else:
                    feat = np.asarray(data[movie][args.modality + '_pooler_output'])
                    last_hidden = np.asarray(data[movie][args.modality + '_last_hidden_state'])
                    feat = np.append(feat, np.reshape(last_hidden, (len(feat), -1)), 1)
                
                features_list.append(feat)
                chunks_per_movie.append(len(feat))
                movie_splits.append(movie)
                
    if not features_list:
        print(f"No train features found for {args.modality}. Skipping PCA.")
        return
        
    features = np.concatenate(features_list, axis=0)
    features = np.nan_to_num(features)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    np.save(out_pca_dir / 'scaler_param.npy', {'mean_': scaler.mean_, 'scale_': scaler.scale_, 'var_': scaler.var_})

    n_components = features.shape[1] if args.modality == 'audio' else min(250, features.shape[1], features.shape[0])
    pca = PCA(n_components=n_components, random_state=seed)
    features = pca.fit_transform(features).astype(np.float32)

    np.save(out_pca_dir / 'pca_param.npy', {
        'components_': pca.components_, 'explained_variance_': pca.explained_variance_,
        'explained_variance_ratio_': pca.explained_variance_ratio_,
        'singular_values_': pca.singular_values_, 'mean_': pca.mean_
    })

    features_train = {}
    count = 0
    for m, movie in enumerate(movie_splits):
        c = chunks_per_movie[m]
        features_train[movie] = features[count:count+c]
        count += c
    np.save(out_pca_dir / 'features_train.npy', features_train)
    print(f"Saved train PCA features to {out_pca_dir / 'features_train.npy'}")


def load_fmri(args, project_root):
    fmri_dir = project_root / 'Data' / 'fmri' / f'sub-{args.subject:02d}' / 'func'
    fmri_file_friends = f'sub-{args.subject:02d}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_file_movie10 = f'sub-{args.subject:02d}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    
    fmri = []
    movie_split_names = []
    movie_split_samples = []

    if not (fmri_dir / fmri_file_friends).exists() or not (fmri_dir / fmri_file_movie10).exists():
        print("fMRI data files not found. Ensure Data/fmri is populated.")
        return None, None, None

    with h5py.File(fmri_dir / fmri_file_friends, 'r') as f:
        for key, val in f.items():
            start, end = args.excluded_samples_start, args.excluded_samples_end
            end_idx = -end if end > 0 else None
            fmri_part = val[start:end_idx]
            fmri.append(fmri_part)
            movie_split_names.append(key[13:])
            movie_split_samples.append(len(fmri_part))

    with h5py.File(fmri_dir / fmri_file_movie10, 'r') as f:
        for key, val in f.items():
            start, end = args.excluded_samples_start, args.excluded_samples_end
            end_idx = -end if end > 0 else None
            fmri_part = f[key][start:end_idx]
            fmri.append(fmri_part)
            
            if key[13:20] == 'figures': name = key[13:22]
            elif key[13:17] == 'life': name = key[13:19]
            else: name = key[13:]
            
            movie_split_names.append(name)
            movie_split_samples.append(len(fmri_part))

    fmri = np.concatenate(fmri, axis=0)
    return fmri, movie_split_names, movie_split_samples


def align_features(args, project_root, out_pca_dir, movie_split_names, movie_split_samples):
    print(f"\n--- Aligning PCA features to fMRI HRF shift for {args.modality} ---")
    pca_file = out_pca_dir / 'features_train.npy'
    if not pca_file.exists():
        print(f"Cannot align: PCA features not found at {pca_file}")
        return None

    features = {args.modality: np.load(pca_file, allow_pickle=True).item()}
    stim_features = []

    for m, split in enumerate(tqdm(movie_split_names, desc="Aligning movies")):
        for s in range(movie_split_samples[m]):
            f_all = np.empty(0)
            mod = args.modality
            
            if split not in features[mod]:
                # If feature is missing, fill with zeros
                sz = args.stimulus_window * features[mod][list(features[mod].keys())[0]].shape[1] if mod in ['visual', 'audio'] else features[mod][list(features[mod].keys())[0]].shape[1]
                f_all = np.append(f_all, np.zeros(sz, dtype=np.float32))
                stim_features.append(f_all)
                continue

            if mod in ['visual', 'audio']:
                if s < (args.stimulus_window + args.hrf_delay):
                    idx_start = args.excluded_samples_start
                    idx_end = idx_start + args.stimulus_window
                else:
                    idx_start = s + args.excluded_samples_start - args.hrf_delay - args.stimulus_window + 1
                    idx_end = idx_start + args.stimulus_window
                
                if idx_end > len(features[mod][split]):
                    idx_end = len(features[mod][split])
                    idx_start = idx_end - args.stimulus_window
                
                f = features[mod][split][idx_start:idx_end]
                f_all = np.append(f_all, f.flatten())

            elif mod == 'language':
                if s < args.hrf_delay:
                    idx = args.excluded_samples_start
                else:
                    idx = s + args.excluded_samples_start - args.hrf_delay
                
                if idx >= (len(features[mod][split]) - args.hrf_delay):
                    f = features[mod][split][-1, :]
                else:
                    if idx < len(features[mod][split]):
                        f = features[mod][split][idx]
                    else:
                        f = np.zeros(features[mod][split].shape[1])
                f_all = np.append(f_all, f.flatten())

            stim_features.append(f_all)

    return np.asarray(stim_features, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, choices=['visual', 'audio', 'language'], default='visual')
    parser.add_argument('--subject', type=int, default=1)
    parser.add_argument('--excluded_samples_start', type=int, default=5)
    parser.add_argument('--excluded_samples_end', type=int, default=5)
    parser.add_argument('--hrf_delay', type=int, default=3)
    parser.add_argument('--stimulus_window', type=int, default=3)
    args = parser.parse_args()

    seed = 20200220
    np.random.seed(seed)
    random.setstate(random.Random(seed).getstate())

    project_root = Path(__file__).resolve().parent.parent.parent
    raw_base_dir = project_root / 'Data' / 'features' / 'raw'
    out_pca_dir = project_root / 'Data' / 'features' / 'pca' / args.modality
    out_pca_dir.mkdir(parents=True, exist_ok=True)
    out_aligned_dir = project_root / 'Data' / 'features' / 'aligned' / f'sub-{args.subject:02d}'
    out_aligned_dir.mkdir(parents=True, exist_ok=True)

    # 1. PCA Downsample
    process_pca(args, seed, raw_base_dir, out_pca_dir)

    # 2. Align features to fMRI target
    fmri, movie_split_names, movie_split_samples = load_fmri(args, project_root)
    if fmri is not None:
        X_aligned = align_features(args, project_root, out_pca_dir, movie_split_names, movie_split_samples)
        if X_aligned is not None:
            np.save(out_aligned_dir / f'X_{args.modality}_train.npy', X_aligned)
            np.save(out_aligned_dir / f'y_fmri_train.npy', fmri)
            print(f"Successfully aligned and saved mapping objects.")
            print(f"X_{args.modality}_train shape: {X_aligned.shape}")
            print(f"y_fmri_train shape: {fmri.shape}")

if __name__ == "__main__":
    main()
