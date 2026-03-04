import os
import glob
from pathlib import Path

import h5py
import numpy as np
import librosa
import string
import pandas as pd
from moviepy import VideoFileClip

import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.transforms import Compose, Lambda, CenterCrop
from pytorchvideo.transforms import Normalize, UniformTemporalSubsample, ShortSideScale

def define_frames_transform(args):
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8

    transform = Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            Normalize(mean, std),
            ShortSideScale(size=side_size),
            CenterCrop(crop_size)
        ]
    )
    return transform

def get_vision_model(args, device):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    model = model.eval()
    model = model.to(device)
    train_nodes, _ = get_graph_node_names(model)
    model_layer = 'blocks.5.pool'
    feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])
    return feature_extractor, model_layer

def get_stimuli_dir(args):
    # Base data dir is project_root / Data
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / 'Data' / 'stimuli'

def list_movie_splits(args):
    stimuli_dir = get_stimuli_dir(args)
    if args.modality == 'language':
        movie_dir = stimuli_dir / 'transcripts' / args.movie_type / args.stimulus_type
        file_type = 'tsv'
    else:
        movie_dir = stimuli_dir / 'movies' / args.movie_type / args.stimulus_type
        file_type = 'mkv'

    if not movie_dir.exists():
        print(f"Directory not found: {movie_dir}")
        return []

    if args.movie_type == 'friends':
        movie_splits_list = [
            Path(x).stem[8:]
            for x in sorted(glob.glob(f"{movie_dir}/*.{file_type}"))
        ]
    elif args.movie_type == 'movie10':
        if args.modality != 'language':
            movie_splits_list = [
                Path(x).stem
                for x in sorted(glob.glob(f"{movie_dir}/*.{file_type}"))
            ]
        else:
            movie_splits_list = [
                Path(x).stem[8:]
                for x in sorted(glob.glob(f"{movie_dir}/*.{file_type}"))
            ]
    return movie_splits_list

def extract_visual_features(args, movie_split, feature_extractor, model_layer, transform, device, save_dir, batch_size=16):
    # Use /dev/shm (RAM disk) for temp files to avoid slow disk I/O
    temp_dir = '/dev/shm/algonauts_visual_temp'
    os.makedirs(temp_dir, exist_ok=True)

    stimuli_dir = get_stimuli_dir(args)
    if args.movie_type == 'friends':
        stim_path = stimuli_dir / 'movies' / args.movie_type / args.stimulus_type / f'friends_{movie_split}.mkv'
    elif args.movie_type == 'movie10':
        stim_path = stimuli_dir / 'movies' / args.movie_type / args.stimulus_type / f'{movie_split}.mkv'

    clip = VideoFileClip(str(stim_path))
    start_times = [x for x in np.arange(0, clip.duration, args.tr)][:-1]

    # Collect all transformed frame tensors first
    chunk_tensors = []
    for start in start_times:
        clip_chunk = clip.subclipped(start, start + args.tr)
        chunk_path = os.path.join(temp_dir, f'visual_{args.stimulus_type}.mp4')
        clip_chunk.write_videofile(chunk_path, logger=None)

        video_clip = VideoFileClip(chunk_path)
        chunk_frames = [f for f in video_clip.iter_frames()]
        frames_array = np.transpose(chunk_frames, [3, 0, 1, 2])

        inputs = torch.from_numpy(frames_array)
        inputs = transform(inputs)  # [C, T, H, W]
        chunk_tensors.append(inputs)

    # Batched GPU inference
    visual_features = []
    for i in range(0, len(chunk_tensors), batch_size):
        batch = torch.stack(chunk_tensors[i:i+batch_size]).to(device)  # [B, C, T, H, W]
        with torch.no_grad():
            preds = feature_extractor(batch)
        for feat in preds[model_layer].cpu().numpy():
            visual_features.append(feat.reshape(-1))

    visual_features = np.array(visual_features, dtype='float32')
    out_file = os.path.join(save_dir, f'{args.movie_type}_{args.stimulus_type}_features_visual.h5')
    flag = 'a' if Path(out_file).exists() else 'w'
    with h5py.File(out_file, flag) as f:
        group = f.require_group(movie_split)
        if 'visual' in group:
            del group['visual']
        group.create_dataset('visual', data=visual_features, dtype=np.float32)

def extract_audio_features(args, movie_split, device, save_dir):
    # Use /dev/shm (RAM disk) for temp files to avoid slow disk I/O
    temp_dir = '/dev/shm/algonauts_audio_temp'
    os.makedirs(temp_dir, exist_ok=True)

    stimuli_dir = get_stimuli_dir(args)
    if args.movie_type == 'friends':
        stim_path = stimuli_dir / 'movies' / args.movie_type / args.stimulus_type / f'friends_{movie_split}.mkv'
    elif args.movie_type == 'movie10':
        stim_path = stimuli_dir / 'movies' / args.movie_type / args.stimulus_type / f'{movie_split}.mkv'

    clip = VideoFileClip(str(stim_path))
    start_times = [x for x in np.arange(0, clip.duration, args.tr)][:-1]

    audio_features = []
    for start in start_times:
        clip_chunk = clip.subclipped(start, start + args.tr)
        chunk_path = os.path.join(temp_dir, f'audio_{args.stimulus_type}.wav')
        clip_chunk.audio.write_audiofile(chunk_path, logger=None)

        y, sr = librosa.load(chunk_path, sr=args.sr, mono=True)
        audio_features.append(np.mean(librosa.feature.mfcc(y=y, sr=sr), 1))

    audio_features = np.array(audio_features, dtype='float32')

    out_file = os.path.join(save_dir, f'{args.movie_type}_{args.stimulus_type}_features_audio.h5')
    flag = 'a' if Path(out_file).exists() else 'w'
    with h5py.File(out_file, flag) as f:
        group = f.require_group(movie_split)
        if 'audio' in group:
            del group['audio']
        group.create_dataset('audio', data=audio_features, dtype=np.float32)

def extract_language_features(args, movie_split, model, tokenizer, device, save_dir, batch_size=32):
    stimuli_dir = get_stimuli_dir(args)
    if args.movie_type == 'friends':
        stim_path = stimuli_dir / 'transcripts' / args.movie_type / args.stimulus_type / f'friends_{movie_split}.tsv'
    elif args.movie_type == 'movie10':
        stim_path = stimuli_dir / 'transcripts' / args.movie_type / args.stimulus_type / f'movie10_{movie_split}.tsv'

    df = pd.read_csv(stim_path, sep='\t')
    df.insert(loc=0, column="is_na", value=df["text_per_tr"].isna())

    tokens = []
    np_tokens = []
    # Accumulate all token sequences per TR first
    pooler_input_ids_list = []
    last_hidden_input_ids_list = []
    has_tokens_list = []
    has_np_tokens_list = []

    for i in range(df.shape[0]):
        if not df.iloc[i]["is_na"]:
            tr_text = str(df.iloc[i]["text_per_tr"])
            tokens.extend(tokenizer.tokenize(tr_text))
            tr_np_tokens = tokenizer.tokenize(tr_text.translate(str.maketrans('', '', string.punctuation)))
            np_tokens.extend(tr_np_tokens)

        if len(tokens) > 0:
            used = tokenizer.convert_tokens_to_ids(tokens[-(args.num_used_tokens):])
            pooler_input_ids_list.append([101] + used + [102])
            has_tokens_list.append(True)
        else:
            pooler_input_ids_list.append(None)
            has_tokens_list.append(False)

        if len(np_tokens) > 0:
            used_np = tokenizer.convert_tokens_to_ids(np_tokens[-(args.num_used_tokens):])
            last_hidden_input_ids_list.append([101] + used_np + [102])
            has_np_tokens_list.append(True)
        else:
            last_hidden_input_ids_list.append(None)
            has_np_tokens_list.append(False)

    def pad_batch(ids_list):
        """Pad a list of token id sequences to the same length."""
        max_len = max(len(x) for x in ids_list)
        padded = [x + [0] * (max_len - len(x)) for x in ids_list]
        attention = [[1]*len(x) + [0]*(max_len - len(x)) for x in ids_list]
        return torch.tensor(padded), torch.tensor(attention)

    # Batched pooler_output inference
    pooler_output_results = [None] * len(pooler_input_ids_list)
    valid_pooler_idx = [i for i, h in enumerate(has_tokens_list) if h]
    for start in range(0, len(valid_pooler_idx), batch_size):
        batch_idx = valid_pooler_idx[start:start+batch_size]
        ids_batch = [pooler_input_ids_list[i] for i in batch_idx]
        input_tensor, attn_mask = pad_batch(ids_batch)
        input_tensor = input_tensor.to(device)
        attn_mask = attn_mask.to(device)
        with torch.no_grad():
            outputs = model(input_tensor, attention_mask=attn_mask)
        for j, i in enumerate(batch_idx):
            pooler_output_results[i] = outputs['pooler_output'][j].cpu().numpy()

    # Batched last_hidden_state inference
    last_hidden_results = [None] * len(last_hidden_input_ids_list)
    valid_lhs_idx = [i for i, h in enumerate(has_np_tokens_list) if h]
    for start in range(0, len(valid_lhs_idx), batch_size):
        batch_idx = valid_lhs_idx[start:start+batch_size]
        ids_batch = [last_hidden_input_ids_list[i] for i in batch_idx]
        input_tensor, attn_mask = pad_batch(ids_batch)
        input_tensor = input_tensor.to(device)
        attn_mask = attn_mask.to(device)
        with torch.no_grad():
            outputs = model(input_tensor, attention_mask=attn_mask)
        for j, i in enumerate(batch_idx):
            np_outputs = outputs['last_hidden_state'][j][1:-1].cpu().numpy()
            np_feat = np.full((args.kept_tokens_last_hidden_state, 768), np.nan, dtype='float32')
            # Only use non-padding tokens
            real_token_count = attn_mask[j].sum().item() - 2  # minus [CLS] and [SEP]
            tk_idx = min(args.kept_tokens_last_hidden_state, int(real_token_count))
            if tk_idx > 0:
                np_feat[-tk_idx:, :] = np_outputs[-tk_idx:]
            last_hidden_results[i] = np_feat

    # Assemble final arrays
    pooler_output = np.array(
        [pooler_output_results[i] if has_tokens_list[i] else np.full(768, np.nan, dtype='float32')
         for i in range(len(pooler_input_ids_list))], dtype='float32')
    last_hidden_state = np.array(
        [last_hidden_results[i] if has_np_tokens_list[i] else np.full((args.kept_tokens_last_hidden_state, 768), np.nan, dtype='float32')
         for i in range(len(last_hidden_input_ids_list))], dtype='float32')

    out_file = os.path.join(save_dir, f'{args.movie_type}_{args.stimulus_type}_features_language.h5')
    flag = 'a' if Path(out_file).exists() else 'w'
    with h5py.File(out_file, flag) as f:
        group = f.require_group(movie_split)
        if 'language_pooler_output' in group: del group['language_pooler_output']
        if 'language_last_hidden_state' in group: del group['language_last_hidden_state']
        group.create_dataset('language_pooler_output', data=pooler_output, dtype=np.float32)
        group.create_dataset('language_last_hidden_state', data=last_hidden_state, dtype=np.float32)
