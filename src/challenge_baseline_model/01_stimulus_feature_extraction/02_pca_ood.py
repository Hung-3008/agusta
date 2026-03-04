"""Downsample the OOD movies stimulus features using PCA.

Parameters
----------
modality : str
	Whether to use 'visual', 'audio' or 'language' features.
project_dir : str
	Directory of the Algonauts 2025 folder.

"""

import argparse
import os
import numpy as np
import h5py
import random
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--modality', type=str, default='visual')
parser.add_argument('--project_dir', default='../algonauts_2025/', type=str)
args = parser.parse_args()

print('>>> Stimulus features PCA OOD <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)
random.seed(seed)


# =============================================================================
# Output directory
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
	'pca', 'ood', args.modality)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)


# =============================================================================
# OOD stimulus features
# =============================================================================
if args.modality != 'language':
	movies = ['chaplin', 'mononoke', 'passepartout', 'planetearth',
		'pulpfiction', 'wot']
else:
	movies = ['mononoke', 'passepartout', 'planetearth',
		'pulpfiction', 'wot']

# Load the stimulus features for the OOD movies
episode_names = []
chunks_per_episode = []
for movie in tqdm(movies):
	data_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
		'raw', 'ood', args.modality, 'ood_'+movie+'_features_'+args.modality+
		'.h5')
	data = h5py.File(data_dir, 'r')
	for e, episode in enumerate(data.keys()):
		if movie == movies[0] and e == 0: # if first episode of first movie
			if args.modality != 'language':
				features = np.asarray(data[episode][args.modality])
			else:
				features = np.asarray(
					data[episode][args.modality+'_pooler_output'])
				features = np.append(features, np.asarray(np.reshape(
					data[episode][args.modality+'_last_hidden_state'],
					(len(features), -1))), 1)
		else:
			if args.modality != 'language':
				features = np.append(
					features, np.asarray(data[episode][args.modality]), 0)
			else:
				feat = np.asarray(
					data[episode][args.modality+'_pooler_output'])
				feat = np.append(feat, np.asarray(np.reshape(
					data[episode][args.modality+'_last_hidden_state'],
					(len(feat), -1))), 1)
				features = np.append(features, feat, 0)
		if args.modality != 'language':
			chunks_per_episode.append(len(data[episode][args.modality]))
		else:
			chunks_per_episode.append(len(
				data[episode][args.modality+'_pooler_output']))
		episode_names.append(episode)
	del data

# Convert NaN values to zeros (PCA doesn't accept NaN values)
features = np.nan_to_num(features)

# z-score the features
scaler_param = np.load(os.path.join(args.project_dir, 'results',
	'stimulus_features', 'pca', 'friends_movie10', args.modality,
	'scaler_param.npy'), allow_pickle=True).item()
scaler = StandardScaler()
scaler.mean_ = scaler_param['mean_']
scaler.scale_ = scaler_param['scale_']
scaler.var_ = scaler_param['var_']
features = scaler.transform(features)
del scaler, scaler_param

# Downsample the features using PCA
pca_param = np.load(os.path.join(args.project_dir, 'results',
	'stimulus_features', 'pca', 'friends_movie10', args.modality,
	'pca_param.npy'), allow_pickle=True).item()
if args.modality == 'audio':
	n_components = features.shape[1]
else:
	n_components = 250
pca = PCA(n_components=n_components, random_state=seed)
pca.components_ = pca_param['components_']
pca.explained_variance_ = pca_param['explained_variance_']
pca.explained_variance_ratio_ = pca_param['explained_variance_ratio_']
pca.singular_values_ = pca_param['singular_values_']
pca.mean_ = pca_param['mean_']
features = pca.transform(features)
features = features.astype(np.float32)
del pca, pca_param

# Convert the features to float32
features = features.astype(np.float32)

# Reshape the features into individual episodes
features_test = {}
count = 0
for e, epi in enumerate(episode_names):
	chunks = chunks_per_episode[e]
	features_test[epi] = features[count:count+chunks]
	count += chunks
del features

# Save the features
data = np.save(os.path.join(save_dir, 'features_ood.npy'), features_test)
