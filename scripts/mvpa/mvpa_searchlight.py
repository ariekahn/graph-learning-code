import os
import logging
import argparse
import pandas as pd
import numpy as np
from nilearn import decoding
from nilearn.image import load_img, math_img, new_img_like, concat_imgs, index_img
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut

parser = argparse.ArgumentParser('Run MVPA analysis')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('project_dir', type=str, help='directory containing data and derived subfolders')
parser.add_argument('output_dir', type=str, help='Location for output files')
parser.add_argument('n_threads', type=int, help='Number of simultaneous searchlights')
args = parser.parse_args()

subject_id = args.subject_id
project_dir = args.project_dir
output_dir = args.output_dir
n_threads = args.n_threads

os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/images', exist_ok=True)

logging.basicConfig(filename=f'{output_dir}/sub-{subject_id}.log', filemode='w', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info(f'Subject: {subject_id}')

fmriprep_dir = f'{project_dir}/derived/fmriprep'
lss_dir = f'{project_dir}/derived/feat_representation_lss'
localizer_dir = f'{project_dir}/derived/feat_localizer'
data_dir = f'{project_dir}/data'

# Create some helper functions for
# 1. Loading the T1 template
# 2. Loading the LOC template


def load_masked_template(subject_id):
    """
    Load the T1w template, and mask it

    derived/fmriprep/sub-{subject_id}/anat/sub-{subject_id}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz

    Usage:
    template = load_masked_template('GLS011')
    """
    # Load T1w template and mask
    template = load_img(f'{fmriprep_dir}/sub-{subject_id}/anat/sub-{subject_id}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz')
    template_mask = load_img(f'{fmriprep_dir}/sub-{subject_id}/anat/sub-{subject_id}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
    template = math_img("img1 * img2", img1=template, img2=template_mask)
    return template


def load_benson_mask(subject_id):
    varea = load_img(f'{project_dir}/derived/template_benson/benson14_varea/sub-{subject_id}/sub-{subject_id}_ses-2_task-graphrepresentation_run-1_space-MNI152NLin2009cAsym_benson14_varea.nii.gz')
    mask = math_img('img > 0', img=varea)
    return mask


def load_aparc(subject_id):
    """
    Load Freesurfer aparc for localizer

    derived/fmriprep/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz
    """
    aparc = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz')
    return aparc


def load_loc(subject_id, hemi='both'):
    """
    Load a mask for LOC

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-lateraloccipital = 1
    ctx-rh-lateraloccipital = 2

    We want 2011 ctx-rh-lateraloccipital and 1011 ctx-lh-lateraloccipital
    """
    roi_str = ''
    if hemi == 'left':
        roi_str = 'img == 1011'
    elif hemi == 'right':
        roi_str = 'img == 2011'
    elif hemi == 'both':
        roi_str = '(img == 1011) | (img == 2011)'
    elif hemi == 'separate':
        roi_str = '(img == 1011) + 2*(img == 2011)'
    else:
        raise ValueError('hemi must be left/right/both/separate')

    aparc = load_aparc(subject_id)
    loc = math_img(roi_str, img=aparc)
    return loc


def load_entorhinal(subject_id, hemi='both'):
    """
    Load a mask for Entorhinal Cortex

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-??? = 1
    ctx-rh-??? = 2

    We want 2011 ctx-rh-lateraloccipital and 1011 ctx-lh-lateraloccipital
    """
    roi_str = ''
    if hemi == 'left':
        roi_str = 'img == 1006'
    elif hemi == 'right':
        roi_str = 'img == 2006'
    elif hemi == 'both':
        roi_str = '(img == 1006) | (img == 2006)'
    elif hemi == 'separate':
        roi_str = '(img == 1006) + 2*(img == 2006)'
    else:
        raise ValueError('hemi must be left/right/both/separate')

    aparc = load_aparc(subject_id)
    entorhinal = math_img(roi_str, img=aparc)
    return entorhinal


def load_hippocampus(subject_id, hemi='both'):
    """
    Load a mask for Hippocampus

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-??? = 1
    ctx-rh-??? = 2

    We want 2011 ctx-rh-lateraloccipital and 1011 ctx-lh-lateraloccipital
    """
    roi_str = ''
    if hemi == 'left':
        roi_str = 'img == 17'
    elif hemi == 'right':
        roi_str = 'img == 53'
    elif hemi == 'both':
        roi_str = '(img == 17) | (img == 53)'
    elif hemi == 'separate':
        roi_str = '(img == 17) + 2*(img == 53)'
    else:
        raise ValueError('hemi must be left/right/both/separate')

    aparc = load_aparc(subject_id)
    hippocampus = math_img(roi_str, img=aparc)
    return hippocampus


def load_zstat_index(subject_id):
    """
    Loads zstat index where thresholded cluster voxels are labeled
    derived/localizer/clusters/sub-{subject_id}/fwhm-5.0/zstat1_index.nii.gz
    """
    img = load_img(f'{localizer_dir}/sub-{subject_id}/clusters/fwhm-5.0_zstat1_index.nii.gz')
    return img


def load_zstat(subject_id):
    """
    Loads zstat statistical map
    derived/localizer/zfiles/sub-{subject_id}/fwhm-5.0/zstat1_index.nii.gz
    """
    img = load_img(f'{localizer_dir}/sub-{subject_id}/zfiles/fwhm-5.0_zstat1.nii.gz')
    return img


def load_bold_mask(subject_id):
    """
    Load bold mask intersection across all runs

    derived/fmriprep/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
    """
    mask1 = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
    mask2 = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-2_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
    mask3 = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-3_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
    mask4 = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-4_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
    mask5 = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-5_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
    mask6 = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-6_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
    mask7 = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-7_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
    mask8 = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-8_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')

    boldmask = math_img("mask1 * mask2 * mask3 * mask4 * mask5 * mask6 * mask7 * mask8",
                        mask1=mask1, mask2=mask2, mask3=mask3, mask4=mask4,
                        mask5=mask5, mask6=mask6, mask7=mask7, mask8=mask8)
    return boldmask


# Skip pe 0, not included
n_excluded_pes = 1
n_pes = 60 - n_excluded_pes


def load_pes_lss_volume(subtract_mean=True, rescale=True):
    """LS-S regression with one regressor for each event

       Run a separate GLM for each event

    """
    pes = []
    for run in range(1, 9):
        pes_run = []
        for event in range(n_excluded_pes, 60):
            pe = load_img(f'{lss_dir}/sub-{subject_id}/run-{run}/parameter_estimates/fwhm-5.0_event-{event}_pe1.nii.gz')
            pes_run.append(pe)
        pes_run = concat_imgs(pes_run)
        pes.append(pes_run)

    if subtract_mean:
        # Find and subtract the mean of each voxel, for each run
        # Take the mean over events, then re-add a single dimension, and subtract
        for run in range(8):
            data = pes[run].get_fdata()
            mean_data = np.expand_dims(data.mean(3), 3)
            pes[run] = new_img_like(pes[run], data - mean_data)

    if rescale:
        # Rescale each voxel to 1
        for run in range(8):
            data = pes[run].get_fdata()
            std_data = np.expand_dims(data.std(3), 3)
            std_data[std_data == 0] = 1  # Avoid divide by 0 for constant voxels
            pes[run] = new_img_like(pes[run], data / std_data)

    pes = concat_imgs(pes)

    if np.isnan(pes.get_fdata()).any():
        raise ValueError

    return pes


logging.info('Loading PEs')
pes_lss = load_pes_lss_volume()
logging.info('pes LS-S:')
logging.info(f'{pes_lss.shape}')

f, ax = plt.subplots(ncols=2, figsize=(10, 4))
ax[0].plot(pes_lss.get_fdata()[50, 50, 50, :])
ax[0].set_title('Example Voxel')
ax[0].set_xlabel('Trial')
ax[0].set_ylabel('Amplitude')

ax[1].plot(pes_lss.get_fdata()[50, :, 50, 0])
ax[1].set_title('Example Trial')
ax[1].set_xlabel('Voxel')
ax[1].set_ylabel('Amplitude')
f.suptitle(f'{subject_id}')
f.tight_layout()
f.savefig(f'{output_dir}/images/sub-{subject_id}_sample_pes.pdf')

# Nodes and blocks
logging.info('Loading nodes and blocks')

events = []
for run in range(1, 9):
    filename = f'{data_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-{run}_events.tsv'
    data = pd.read_csv(filename, sep='\t')
    data['steadystate'] = True
    data.loc[:n_excluded_pes - 1, 'steadystate'] = False
    data['block'] = run - 1
    events.append(data)
events = pd.concat(events).reset_index(drop=True)
nodes_lss = events[events['steadystate']]['node'].values
nodes_lsa = np.tile(range(15), 8)

blocks_lss = events[events['steadystate']]['block'].values
blocks_lsa = np.repeat(range(8), 15)

f, ax = plt.subplots(2, 2)
ax[0, 0].plot(blocks_lss, label='LS-S')
ax[0, 0].set_title('Blocks LS-S')
ax[1, 0].plot(blocks_lsa, label='LS-A', color='orange')
ax[1, 0].set_title('Blocks LS-A')

ax[0, 1].plot(nodes_lss, label='LS-S')
ax[0, 1].set_title('Nodes LS-S')
ax[1, 1].plot(nodes_lsa, label='LS-A', color='orange')
ax[1, 1].set_title('Nodes LS-A')
f.suptitle(f'{subject_id}')
f.tight_layout()
f.savefig(f'{output_dir}/images/sub-{subject_id}_blocks_nodes.pdf')

# Label previous and future events, for prediction
events['node_prev'] = events.groupby('block')['node'].shift(1).astype('Int8')
events['block_prev'] = events.groupby('block')['block'].shift(1).astype('Int8')

events['node_next'] = events.groupby('block')['node'].shift(-1).astype('Int8')
events['block_next'] = events.groupby('block')['block'].shift(-1).astype('Int8')

events['ind'] = range(len(events))
events['pe_ind'] = events.groupby('block')['ind'].shift(1).astype('Int16') - events['block']

f, ax = plt.subplots()
events.loc[1:10, ['node_prev', 'node', 'node_next']].plot(ax=ax)
f.tight_layout()
ax.set_title(f'{subject_id}')
ax.set_xlabel('Trial')
ax.set_ylabel('Node')
f.savefig(f'{output_dir}/images/sub-{subject_id}_shift_nodes.pdf')

cluster_1 = (events['node'] < 5)
cluster_2 = (events['node'] < 10) & (events['node'] > 4)
cluster_3 = (events['node'] < 15) & (events['node'] > 9)
events['cluster'] = 0
events.loc[cluster_1, 'cluster'] = 1
events.loc[cluster_2, 'cluster'] = 2
events.loc[cluster_3, 'cluster'] = 3
events['cluster_prev'] = events.groupby('block')['cluster'].shift(1).astype('Int8')

nodes_temp = events['node'].copy()
for shift_ind in range(5):
    cluster_1 = (nodes_temp < 5)
    cluster_2 = (nodes_temp < 10) & (nodes_temp > 4)
    cluster_3 = (nodes_temp < 15) & (nodes_temp > 9)

    events[f'cluster_shifted_{shift_ind}'] = 0
    events.loc[cluster_1, f'cluster_shifted_{shift_ind}'] = 1
    events.loc[cluster_2, f'cluster_shifted_{shift_ind}'] = 2
    events.loc[cluster_3, f'cluster_shifted_{shift_ind}'] = 3

    nodes_temp += 1
    nodes_temp[nodes_temp == 15] = 0

radii = [7.2, 4.8]  # In mm. 2.4mm = 1 voxel
max_iter = 100000
bold_mask = load_bold_mask(subject_id)
# process_mask_img = load_benson_mask(subject_id)
process_mask_img = load_bold_mask(subject_id)
logging.info('Running Searchlight')

gkf = LeaveOneGroupOut()
clf = LinearSVC(max_iter=max_iter)

classifiers = [
    ('cluster-shifted-0', events.loc[events['steadystate'], 'cluster_shifted_0'].astype(int).values),
    ('cluster-shifted-1', events.loc[events['steadystate'], 'cluster_shifted_1'].astype(int).values),
    ('cluster-shifted-2', events.loc[events['steadystate'], 'cluster_shifted_2'].astype(int).values),
    ('cluster-shifted-3', events.loc[events['steadystate'], 'cluster_shifted_3'].astype(int).values),
    ('cluster-shifted-4', events.loc[events['steadystate'], 'cluster_shifted_4'].astype(int).values),
    ('node', events.loc[events['steadystate'], 'node'].astype(int).values),
]

for radius in radii:
    logging.info(f'{radius=}')
    radius_str = str(radius).replace('.', '-')
    for (classifier_name, y) in classifiers:
        logging.info(f'{radius=} {classifier_name=}')
        searchlight = decoding.SearchLight(mask_img=bold_mask,
                                           process_mask_img=process_mask_img,
                                           radius=radius,
                                           verbose=0,
                                           n_jobs=n_threads,
                                           cv=gkf,
                                           estimator=clf)
        searchlight.fit(pes_lss, y, groups=blocks_lss)
        searchlight_img = new_img_like(pes_lss, searchlight.scores_)
        searchlight_img.to_filename(f'{output_dir}/sub-{subject_id}_searchlight-{radius_str}_{classifier_name}.nii.gz')

    # Previous and Next Nodes
#     for node in range(15):
#         # Trials where the current node is 'node', and the previous node was in the same block
#         prev_trials = events[((events['block'] == events['block_prev'])
#                               & (events['node'] == node))]
#         # Data are the data for those trials
#         trial_inds = prev_trials['pe_ind'].astype(int)
#         data = index_img(pes_lss, trial_inds)
#         # Observations are the labels for the /previous/ trials
#         y = prev_trials['node_prev'].astype(int).values
#         clf = LinearSVC(max_iter=max_iter)
#         loo = LeaveOneOut()
#         searchlight = decoding.SearchLight(mask_img=bold_mask,
#                                            process_mask_img=process_mask_img,
#                                            radius=radius,
#                                            verbose=1,
#                                            n_jobs=n_threads,
#                                            cv=loo,
#                                            estimator=clf)
#         searchlight.fit(data, y)
#         searchlight_img = new_img_like(data, searchlight.scores_)
#         radius_str = str(radius).replace('.', '-')
#         searchlight_img.to_filename(f'{output_dir}/sub-{subject_id}_searchlight-prev_node-{node}_radius-{radius_str}.nii.gz')
#
#     for node in range(15):
#         # Trials where the current node is 'node', and the next node is in the same block
#         # We also ensure the previous block was the same, to not use the first PE
#         next_trials = events[((events['block'] == events['block_next'])
#                               & (events['block'] == events['block_prev'])
#                               & (events['node'] == node))]
#         # Data are the data for those trials
#         trial_inds = next_trials['pe_ind'].astype(int)
#         data = index_img(pes_lss, trial_inds)
#         # Observations are the labels for the /next/ trials
#         y = next_trials['node_next'].astype(int).values
#         clf = LinearSVC(max_iter=max_iter)
#         loo = LeaveOneOut()
#         searchlight = decoding.SearchLight(mask_img=bold_mask,
#                                            process_mask_img=process_mask_img,
#                                            radius=radius,
#                                            verbose=1,
#                                            n_jobs=n_threads,
#                                            cv=loo,
#                                            estimator=clf)
#         searchlight.fit(data, y)
#         searchlight_img = new_img_like(data, searchlight.scores_)
#         radius_str = str(radius).replace('.', '-')
#         searchlight_img.to_filename(f'{output_dir}/sub-{subject_id}_searchlight-next_node-{node}_radius-{radius_str}.nii.gz')
