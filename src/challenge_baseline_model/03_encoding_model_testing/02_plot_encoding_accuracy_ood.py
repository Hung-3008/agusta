"""Plot the encoding models encoding accuracy for the OOD movies.

Parameters
----------
subjects : list
	List of used CNeuroMod subjects.
modalities : list
	List of modalities used for the encoding models training.
ood_movies : list
	List of OOD movies.
project_dir : str
	Directory of the Algonauts 2025 folder.

"""

import argparse
import os
import numpy as np
from nilearn import plotting
from tqdm import tqdm
from nilearn.maskers import NiftiLabelsMasker
import matplotlib
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 5])
parser.add_argument('--modalities', type=list, default=['language']) # ['all', 'audio', 'visual', 'language']
parser.add_argument('--ood_movies', type=list, default=['chaplin', 'mononoke', 'passepartout', 'planetearth', 'pulpfiction', 'wot'])
parser.add_argument('--project_dir', default='../algonauts_2025/', type=str)
args = parser.parse_args()


# =============================================================================
# Saving directory
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results',
	'encoding_accuracy_ood_plots')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)


# =============================================================================
# Loop across modalities
# =============================================================================
for mod in tqdm(args.modalities):


# =============================================================================
# Load the encoding accuracy results
# =============================================================================
	data_dir = os.path.join(args.project_dir, 'results',
		'encoding_accuracy', 'encoding_accuracy_ood_modality-'+mod+'.npy')
	encoding_accuracy = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Average across OOD movies and subjects
# =============================================================================
	avg_all_movies = []
	avg_single_movies = {}
	for m in args.ood_movies:
		if mod in ['language'] and m in ['chaplin']:
			pass
		else:
			avg_all = []
			avg_single = []
			for s in args.subjects:
				avg_all.append(encoding_accuracy['sub-0'+str(s)][m])
				avg_single.append(encoding_accuracy['sub-0'+str(s)][m])
			avg_all_movies.append(np.mean(avg_all, 0))
			avg_single_movies[m] = np.mean(avg_all, 0)
	avg_all_movies = np.mean(avg_all_movies, 0)

	mean_acc = []
	for s in args.subjects:
		acc  = []
		for m in args.ood_movies:
			if mod in ['language'] and m in ['chaplin']:
				pass
			else:
				acc.append(np.mean(encoding_accuracy['sub-0'+str(s)][m]))
		mean_acc.append(np.mean(acc, 0))
	mean_acc = np.round(np.mean(mean_acc, 0), 4)


# =============================================================================
# Plot the movie- and subject-average encoding accuracy
# =============================================================================
	# Load the brain volume atlas
	atlas_file = 'sub-0' + str(1) + \
		'_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_' + \
		'desc-dseg_parcellation.nii.gz'
	atlas_path = os.path.join(args.project_dir, 'data',
		'algonauts_2025.competitors', 'fmri', 'sub-0'+str(1), 'atlas',
		atlas_file)

	# Map the prediction accuracy onto a 3D brain atlas for plotting
	atlas_masker = NiftiLabelsMasker(labels_img=atlas_path)
	atlas_masker.fit()
	nii_file = atlas_masker.inverse_transform(avg_all_movies)

	# Plot the encoding accuracy
	out_file_name = 'encoding_accuracy_ood_movie-avg_modality-' + mod
	title = 'Encoding accuracy OOD, movie-avg, modality-' + mod + \
		', mean accuracy: ' + str(mean_acc)
	display = plotting.plot_glass_brain(
		stat_map_img=nii_file,
#		output_file=os.path.join(save_dir, out_file_name),
		display_mode="lyrz",
		colorbar=True,
		title=title,
		threshold='auto', # 'auto' --> gives bigger brain plots
		cmap='hot_r',
		vmin=0,
		vmax=0.5,
		plot_abs=False,
		symmetric_cbar=False
	)
	# Colorbar
	colorbar = display._cbar
	colorbar.set_label("Pearson's $r$", rotation=90, labelpad=12,
		fontsize=12)
	plotting.show()
	# Save
	display.savefig(os.path.join(save_dir, out_file_name), dpi=300)


# =============================================================================
# Plot the Subject-average encoding accuracy
# =============================================================================
	# Loop across movies
	for m in args.ood_movies:

		if mod in ['language'] and m in ['chaplin']:
			pass
		else:

			# Map the prediction accuracy onto a 3D brain atlas for plotting
			atlas_masker = NiftiLabelsMasker(labels_img=atlas_path)
			atlas_masker.fit()
			nii_file = atlas_masker.inverse_transform(avg_single_movies[m])

			# Plot the encoding accuracy
			mean_acc = np.round(np.mean(avg_single_movies[m]), 4)
			out_file_name = 'encoding_accuracy_ood_movie-' + m + \
			    '_modality-' + mod
			title = 'Encoding accuracy OOD, movie-' + m + ', modality-' + \
			    mod + ', mean accuracy: ' + str(mean_acc)
			display = plotting.plot_glass_brain(
				stat_map_img=nii_file,
	#			output_file=os.path.join(save_dir, out_file_name),
				display_mode="lyrz",
				colorbar=True,
				title=title,
				threshold='auto', # 'auto' --> gives bigger brain plots
				cmap='hot_r',
				vmin=0,
				vmax=0.5,
				plot_abs=False,
				symmetric_cbar=False
			)
			# Colorbar
			colorbar = display._cbar
			colorbar.set_label("Pearson's $r$", rotation=90, labelpad=12,
				fontsize=12)
			plotting.show()
			# Save
			display.savefig(os.path.join(save_dir, out_file_name), dpi=300)
