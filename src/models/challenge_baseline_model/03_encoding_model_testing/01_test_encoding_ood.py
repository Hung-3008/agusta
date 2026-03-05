"""Test the trained encoding models on the OOD movies. Testing includes using
the trained encoding models to predict fMRI responses for the OOD movies, and
then correlating (Pearson's r) these predictions with the corresponding
recorded fMRI responses.

Parameters
----------
subjecs : list
	List of subjects used for testing.
ood_movies : list
	List of OOD movies.
modality : str
	String indicating the stimulus feature modality used to train and test the
	encoding model. Available modalities are ["visual", "audio", "language",
	"all"].
hrf_delay : int
	fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
	that reflects changes in blood oxygenation levels in response to activity
	in the brain. Blood flow increases to a given brain region in response its
	activity. This vascular response, which follows the hemodynamic response
	function (HRF), takes time. Typically, the HRF peaks around 5–6 seconds
	after a neural event: this delay reflects the time needed for blood
	oxygenation changes to propagate and for the fMRI signal to capture them.
	Therefore, this parameter introduces a delay between stimulus samples and
	fMRI samples. For example, with a hrf_delay of 3, if the fMRI sample of
	interest is 20, the corresponding stimulus sample will be 17.
stimulus_window : int
	Integer indicating how many stimulus feature samples are used to model each
	fMRI sample, starting from the stimulus sample corresponding to the fMRI
	sample of interest, minus the hrf_delay, and going back in time. For
	example, with a 'stimulus_window' of 5, if the fMRI sample of interest is
	20, it will be modeled with stimulus samples [16, 17, 18, 19, 20]. Note
	that this only applies to visual and audio features, since the language
	features were already extracted using transcript words spanning several
	movie samples (thus, each fMRI sample will only be modeled using the
	corresponding language feature sample). Also note that a larger stimulus
	window will increase compute time, since it increases the amount of
	stimulus features used to train and test the fMRI encoding models.
project_dir : str
	Directory of the Algonauts 2025 folder.

"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

from test_encoding_ood_utils import load_trained_model
from test_encoding_ood_utils import load_fmri_samples
from test_encoding_ood_utils import load_stimulus_features
from test_encoding_ood_utils import load_fmri

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 5])
parser.add_argument('--ood_movies', type=list, default=['chaplin', 'mononoke', 'passepartout', 'planetearth', 'pulpfiction', 'wot'])
parser.add_argument('--modality', type=str, default='all')
parser.add_argument('--hrf_delay', type=int, default=3)
parser.add_argument('--stimulus_window', type=int, default=5)
parser.add_argument('--project_dir', default='../algonauts_2025/', type=str)

args = parser.parse_args()

print('>>> Test encoding models OOD <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Loop over subjects
# =============================================================================
fmri_test_pred = {}
encoding_accuracy = {}

for sub in tqdm(args.subjects):


# =============================================================================
# Load the trained encoding model
# =============================================================================
	model, model_chaplin = load_trained_model(args, sub)


# =============================================================================
# Load the fMRI sample number for the OOD movies
# =============================================================================
	fmri_samples = load_fmri_samples(args, sub)


# =============================================================================
# Load the stimulus features for the OOD movies
# => regression testing predictors (X)
# =============================================================================
	X_test = load_stimulus_features(args, sub, fmri_samples)


# =============================================================================
# Predict the fMRI responses for the OOD movies
# =============================================================================
	fmri_test_pred['sub-0'+str(sub)] = {}
	for key, val in X_test.items():
		if args.modality in ['all'] and key in ['chaplin1', 'chaplin2']:
			fmri_test_pred['sub-0'+str(sub)][key] = model_chaplin.predict(val).astype(np.float32)
		elif args.modality in ['language'] and key in ['chaplin1', 'chaplin2']:
			pass
		else:
			fmri_test_pred['sub-0'+str(sub)][key] = model.predict(val).astype(np.float32)


# =============================================================================
# Load the ground truth fMRI responses for the OOD movies
# => regression testing targets (y_test)
# =============================================================================
	y_test = load_fmri(args, sub)


# =============================================================================
# Compute the encoding accuracy
# =============================================================================
	encoding_accuracy['sub-0'+str(sub)] = {}
	movies = ['chaplin', 'mononoke', 'passepartout', 'planetearth',
		'pulpfiction', 'wot']

	for m in movies:

		if args.modality in ['language'] and m in ['chaplin']:
			pass

		else:

			# Append the fMRI responses for each OOD movie, while removing the
			# first 5 and last 5 samples of each movie split
			y_test_array = []
			y_test_pred_array = []
			y_test_array.append(y_test[m+'1'][5:-5])
			y_test_array.append(y_test[m+'2'][5:-5])
			y_test_pred_array.append(
				fmri_test_pred['sub-0'+str(sub)][m+'1'][5:-5])
			y_test_pred_array.append(
				fmri_test_pred['sub-0'+str(sub)][m+'2'][5:-5])
			y_test_array = np.concatenate(y_test_array, 0)
			y_test_pred_array = np.concatenate(y_test_pred_array, 0)

			# Correlate the recorded and and predicted fMRI responses at each
			# parcel
			correlation = np.zeros((y_test_array.shape[1]), dtype=np.float32)
			for p in range(len(correlation)):
				correlation[p] = pearsonr(y_test_array[:,p],
					y_test_pred_array[:,p])[0]
			encoding_accuracy['sub-0'+str(sub)][m] = correlation


# =============================================================================
# Save the predicted fMRI responses
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'encoding_predictions')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'encoding_predictions_ood_modality-' + args.modality

np.save(os.path.join(save_dir, file_name), fmri_test_pred)


# =============================================================================
# Save the encoding accuracy
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'encoding_accuracy')

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'encoding_accuracy_ood_modality-' + args.modality

np.save(os.path.join(save_dir, file_name), encoding_accuracy)
