import argparse
import os
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from feature_extraction_utils import (
    define_frames_transform,
    get_vision_model,
    list_movie_splits,
    extract_visual_features,
    extract_audio_features,
    extract_language_features
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--movie_type', type=str, default='movie10')
    parser.add_argument('--stimulus_type', type=str, default='wolf')
    parser.add_argument('--modality', type=str, choices=['visual', 'audio', 'language'], default='language')
    parser.add_argument('--fps', type=float, default=29.97)
    parser.add_argument('--tr', type=float, default=1.49)
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--num_used_tokens', type=int, default=510)
    parser.add_argument('--kept_tokens_last_hidden_state', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for GPU inference (visual: chunks per forward, language: TRs per forward)')
    args = parser.parse_args()

    print('>>> Extract stimulus features <<<')
    print('\nInput parameters:')
    for key, val in vars(args).items():
        print(f'{key:16} {val}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    project_root = Path(__file__).resolve().parent.parent.parent
    save_dir = project_root / 'Data' / 'features' / 'raw' / args.movie_type / args.modality
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.modality == 'visual':
        transform = define_frames_transform(args)
        feature_extractor, model_layer = get_vision_model(args, device)
    elif args.modality == 'language':
        # Load BERT relative to standard transformers
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        model = model.to(device)

    movie_splits_list = list_movie_splits(args)
    if not movie_splits_list:
        print("No movie splits found for criteria. Make sure datalad downloaded the stimuli correctly.")
        return

    print(f"Found {len(movie_splits_list)} movie chunks. Extracting {args.modality} features...")

    for movie_split in tqdm(movie_splits_list):
        if args.modality == 'visual':
            extract_visual_features(args, movie_split, feature_extractor, model_layer, transform, device, str(save_dir), batch_size=args.batch_size)
        elif args.modality == 'audio':
            extract_audio_features(args, movie_split, device, str(save_dir))
        elif args.modality == 'language':
            extract_language_features(args, movie_split, model, tokenizer, device, str(save_dir), batch_size=args.batch_size)

    print(f"Done extracting {args.modality} features.")

if __name__ == "__main__":
    main()
