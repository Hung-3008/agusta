"""Plot the encoding models encoding accuracy for Friends season 7.

Parameters
----------
subjects : list
	List of used CNeuroMod subjects.
modalities : list
	List of modalities used for the encoding models training.
project_dir : str
	Directory of the Algonauts 2025 folder.

"""

import argparse
import numpy as np
import os
from nilearn import plotting
from tqdm import tqdm
from nilearn.maskers import NiftiLabelsMasker
import matplotlib
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--modalities', type=list, default=['all', 'audio', 'visual', 'language'])
parser.add_argument('--project_dir', default='../algonauts_2025/', type=str)
args = parser.parse_args()


# =============================================================================
# Saving directory
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'encoding_accuracy_plots')

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
		'encoding_accuracy', 'encoding_accuracy_friends-s7_modality-'+
		mod+'.npy')
	encoding_accuracy = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Plot the subject-average encoding accuracy
# =============================================================================
	# Append the results across subjects, and average
	avg_encoding_accuracy = []
	mean_acc = []
	for val in encoding_accuracy.values():
		avg_encoding_accuracy.append(val)
		mean_acc.append(np.mean(val))
	avg_encoding_accuracy = np.mean(avg_encoding_accuracy, 0)
	mean_acc = np.round(np.mean(mean_acc), 4)

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
	nii_file = atlas_masker.inverse_transform(avg_encoding_accuracy)

	# Plot the encoding accuracy
	out_file_name = 'encoding_accuracy_friends_s7_sub-avg_modality-' + mod
	title = 'Encoding accuracy Friends s7, sub-avg, modality-' + mod + \
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
# Plot the single-subjects encoding accuracy
# =============================================================================
	for key, val in encoding_accuracy.items():

		# Load the brain volume atlas
		atlas_file = key + \
			'_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_' + \
			'desc-dseg_parcellation.nii.gz'
		atlas_path = os.path.join(args.project_dir, 'data',
			'algonauts_2025.competitors', 'fmri', key, 'atlas', atlas_file)

		# Map the prediction accuracy onto a 3D brain atlas for plotting
		atlas_masker = NiftiLabelsMasker(labels_img=atlas_path)
		atlas_masker.fit()
		nii_file = atlas_masker.inverse_transform(val)

		# Plot the encoding accuracy
		mean_acc = np.round(np.mean(val), 4)
		out_file_name = 'encoding_accuracy_friends_s7_' + key + \
			'_modality-' + mod
		title = 'Encoding accuracy Friends s7, ' + key + ', modality-' + \
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