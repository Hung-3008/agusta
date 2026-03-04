"""Extract OOD movies visual and audio features (from the .mkv movie files), or
language features (from the .tsv movie transcripts).

The visual features are extracted using a 3D ResNet (SlowFast) pre-trained on
action recognition on the Kinetics400 dataset.
https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/

The audio features consist of the Mel-frequency cepstral coefficients (MFCCs)
from the audio signal with librosa.
https://librosa.org/doc/main/generated/librosa.feature.mfcc.html

The language features are extracted using a Bert model.
https://huggingface.co/google-bert/bert-base-uncased

The fMRI volumes were acquired with a repetition time (TR) of 1.49 seconds
(i.e., one volume was acquired every 1.49 seconds). To facilitate pairing
between stimulus features and fMRI data, this code extracts stimulus features
independently for chunks of 1.49 seconds of multimodal movie stimuli.

Parameters
----------
movies : list
	List of OOD movies for which the features are extracted.
modality : str
	Whether to extract 'visual', 'audio' or 'language' features.
fps : float
	Video frames per second.
tr : float
	fMRI repetition time.
sr : int
	Movies audio sampling rate.
num_used_tokens : int
	Total number of tokens that are fed to the language model for each chunk,
	including the tokens from the chunk of interest plus N tokens from previous
	chunks (the maximum allowed by the model is 510).
kept_tokens_last_hidden_state : int
	Number of features retained for the last_hidden_state, where each feature
	corresponds to a token, starting from the most recent token.
project_dir : str
	Directory of the Algonauts 2025 folder.

"""

import argparse
import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from feature_extraction_ood_utils import define_frames_transform
from feature_extraction_ood_utils import get_vision_model
from feature_extraction_ood_utils import list_movie_splits
from feature_extraction_ood_utils import extract_visual_features
from feature_extraction_ood_utils import extract_audio_features
from feature_extraction_ood_utils import extract_language_features

parser = argparse.ArgumentParser()
parser.add_argument('--movies', type=list, default=['chaplin', 'mononoke',
	'passepartout', 'planetearth', 'pulpfiction', 'wot'])
parser.add_argument('--modality', type=str, default='language')
parser.add_argument('--fps', type=float, default=29.97)
parser.add_argument('--tr', type=float, default=1.49)
parser.add_argument('--sr', type=int, default=22050) # original is 44100 for Friends
parser.add_argument('--num_used_tokens', type=int, default=510)
parser.add_argument('--kept_tokens_last_hidden_state', type=int, default=10)
parser.add_argument('--project_dir', default='../algonauts_2025/', type=str)
args = parser.parse_args()

print('>>> Extract stimulus features OOD <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Check for GPU
# =============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Output directory
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
	'raw', 'ood', args.modality)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)


# =============================================================================
# Load the models used for feature extraction
# =============================================================================
if args.modality == 'visual':
	# Load the video DNN transform and model
	transform = define_frames_transform(args)
	feature_extractor, model_layer = get_vision_model(args, device)

elif args.modality == 'language':
	# Load the Bert tokenizer and model
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
		do_lower_case=True)
	model = BertModel.from_pretrained('bert-base-uncased')
	model.eval()
	model = model.to(device)


# =============================================================================
# Loop over movies and episodes
# =============================================================================
for movie in tqdm(args.movies):

	movie_splits_list = list_movie_splits(args, movie)

	for movie_split in tqdm(movie_splits_list):


# =============================================================================
# Extract and save features
# =============================================================================
		if args.modality == 'visual':
			extract_visual_features(
				args,
				movie,
				movie_split,
				feature_extractor,
				model_layer,
				transform,
				device,
				save_dir
				)

		elif args.modality == 'audio':
			extract_audio_features(
				args,
				movie,
				movie_split,
				device,
				save_dir
				)

		elif args.modality == 'language':
			if movie != 'chaplin':
				extract_language_features(
					args,
					movie,
					movie_split,
					model,
					tokenizer,
					device,
					save_dir
					)

