import os
import logging
import argparse
import pandas as pd
import numpy as np
from functools import partial
from scipy.ndimage import binary_dilation
from nilearn import plotting
from nilearn.image import load_img, math_img, new_img_like
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
from sklearn.utils import parallel_backend
from functools import lru_cache
import gc

parser = argparse.ArgumentParser('Run MVPA analysis')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('project_dir', type=str, help='directory containing data and derived subfolders')
parser.add_argument('output_dir', type=str, help='Location for output files')
parser.add_argument('n_threads', type=int, help='Number of threads')
args = parser.parse_args()

subject_id = args.subject_id
project_dir = args.project_dir
output_dir = args.output_dir
n_threads = args.n_threads

os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/images', exist_ok=True)

space = 'MNI152NLin2009cAsym'

logging.basicConfig(filename=f'{output_dir}/mvpa_{subject_id}.log', filemode='w', level=logging.INFO)
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


def load_aparc(subject_id):
    """
    Load Freesurfer aparc for localizer

    derived/fmriprep/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz
    """
    aparc = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz')
    return aparc


def load_aseg(subject_id, hemi='both'):
    """
    Load Freesurfer aparc for localizer

    derived/fmriprep/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz
    """
    assert hemi in ['both', 'lh', 'rh']
    aseg = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-aseg_dseg.nii.gz')
    if hemi == 'lh':
        aseg = math_img('img * (img < 40)', img=aseg)
    elif hemi == 'rh':
        aseg = math_img('img * (img > 40) * (img < 72)', img=aseg)
    return aseg


def load_aparc_roi(left_ind, right_ind, subject_id, hemi):
    roi_str = ''
    if hemi == 'left':
        roi_str = f'img == {left_ind}'
    elif hemi == 'right':
        roi_str = f'img == {right_ind}'
    elif hemi == 'both':
        roi_str = f'(img == {left_ind}) | (img == {right_ind})'
    elif hemi == 'separate':
        roi_str = f'(img == {left_ind}) + 2*(img == {right_ind})'
    else:
        raise ValueError('hemi must be left/right/both/separate')

    aparc = load_aparc(subject_id)
    roi = math_img(roi_str, img=aparc)
    return roi


def load_benson_parc(subject_id):
    """
    Load Benson varea
    """
    varea = load_img(f'{project_dir}/derived/template_benson/benson14_varea/sub-{subject_id}/sub-{subject_id}_ses-2_task-graphrepresentation_run-1_space-MNI152NLin2009cAsym_benson14_varea.nii.gz')
    return varea


def load_benson_roi(subject_id, roi):
    """
    Load Benson varea roi

    roi : integer
        Value of 1-14
    """
    if roi < 1 or roi > 12:
        raise ValueError('ROI must be between 1 and 12')
    varea = load_benson_parc(subject_id)
    mask = math_img(f'img == {roi}', img=varea)
    return mask


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

    We want 1011 ctx-lh-lateraloccipital and 2011 ctx-rh-lateraloccipital
    """
    return load_aparc_roi('1011', '2011', subject_id, hemi)


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
    ctx-lh-entorhinal = 1
    ctx-rh-entorhinal = 2

    We want 1006 ctx-lh-entorhinal and 2006 ctx-rh-entorhinal
    """
    return load_aparc_roi('1006', '2006', subject_id, hemi)


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
    Left-Hippocampus = 1
    Right-Hippocampus = 2

    We want 17 Left-Hippocampus and 53 Right-Hippocampus
    """
    return load_aparc_roi('17', '53', subject_id, hemi)


def load_postcentral(subject_id, hemi='both'):
    """
    Load a mask for Postcentral Gyrus

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-postcentral = 1
    ctx-rh-postcentral = 2

    We want 1022 ctx-lh-postcentral and 2022 ctx-rh-postcentral
    """
    return load_aparc_roi('1022', '2022', subject_id, hemi)


def load_precentral(subject_id, hemi='both'):
    """
    Load a mask for Precentral Gyrus

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-precentral = 1
    ctx-rh-precentral = 2

    We want 1024 ctx-lh-precentral and 2024 ctx-rh-precentral
    """
    return load_aparc_roi('1024', '2024', subject_id, hemi)


def load_pericalcarine(subject_id, hemi='both'):
    """
    Load a mask for Pericalcarine Cortex

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-pericalcarine = 1
    ctx-rh-pericalcarine = 2

    We want 1021 ctx-lh-pericalcarine and 2021 ctx-rh-pericalcarine
    """
    return load_aparc_roi('1021', '2021', subject_id, hemi)


def load_cuneus(subject_id, hemi='both'):
    """
    Load a mask for Cuneus

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-cuneus = 1
    ctx-rh-cuneus = 2

    We want 1005 ctx-lh-cuneus and 2005 ctx-rh-cuneus
    """
    return load_aparc_roi('1005', '2005', subject_id, hemi)


def load_lingual(subject_id, hemi='both'):
    """
    Load a mask for Lingual Cortex

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-lingual = 1
    ctx-rh-lingual = 2

    We want 1013 ctx-lh-lingual and 2013 ctx-rh-lingual
    """
    return load_aparc_roi('1013', '2013', subject_id, hemi)


@lru_cache()
def load_zstat_index(subject_id):
    """
    Loads zstat index where thresholded cluster voxels are labeled
    derived/localizer/clusters/sub-{subject_id}/fwhm-5.0/zstat1_index.nii.gz
    """
    img = load_img(f'{localizer_dir}/sub-{subject_id}/clusters/fwhm-5.0_zstat1_index.nii.gz')
    return img


@lru_cache()
def load_zstat(subject_id):
    """
    Loads zstat statistical map
    derived/localizer/zfiles/sub-{subject_id}/fwhm-5.0/zstat1_index.nii.gz
    """
    img = load_img(f'{localizer_dir}/sub-{subject_id}/zfiles/fwhm-5.0_zstat1.nii.gz')
    return img


# Load T1w template
template = load_masked_template(subject_id)

# Bold Mask
bold_mask = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-{space}_desc-brain_mask.nii.gz')
bold_mask = new_img_like(bold_mask, np.rint(bold_mask.get_fdata()) > 0)

ctx_lateraloccipital = load_loc(subject_id)
zstat_localizer = load_zstat(subject_id)

# Load cluster assignments
cluster_img = load_zstat_index(subject_id)
# We're taking the top 3 clusters, since a few subjects have misidentified clusters
n_clusters = cluster_img.get_fdata().max()
mask_localizer = math_img(f'(img == {n_clusters}) | (img == {n_clusters - 1})', img=cluster_img)
if subject_id in ('GLS024'):
    mask_localizer = math_img(f'(img == {n_clusters}) | (img == {n_clusters - 2})', img=cluster_img)
elif subject_id in ('GLS017'):
    mask_localizer = math_img(f'(img == {n_clusters - 1}) | (img == {n_clusters - 2})', img=cluster_img)

mask_lh = math_img('img > 0', img=load_aseg(subject_id, 'lh'))
mask_rh = math_img('img > 0', img=load_aseg(subject_id, 'rh'))

mask_loc = math_img('img1 & img2', img1=mask_localizer, img2=ctx_lateraloccipital)
mask_loc_lh = math_img('img1 & img2 & img3', img1=mask_localizer, img2=ctx_lateraloccipital, img3=mask_lh)
mask_loc_rh = math_img('img1 & img2 & img3', img1=mask_localizer, img2=ctx_lateraloccipital, img3=mask_rh)

# Now make sure we have enough voxels. If not, dilate the mask
n_voxels_overlap = mask_loc.get_fdata().sum()
while n_voxels_overlap < 800:
    logging.info(f'Only {n_voxels_overlap} in overlap. Dilating.')
    mask_loc = new_img_like(mask_loc, binary_dilation(mask_loc.get_fdata(), mask=bold_mask.get_fdata()))
    n_voxels_overlap = mask_loc.get_fdata().sum()

n_voxels_overlap_lh = mask_loc_lh.get_fdata().sum()
while n_voxels_overlap_lh < 200:
    logging.info(f'Only {n_voxels_overlap_lh} in overlap_lh. Dilating.')
    mask_loc_lh = new_img_like(mask_loc_lh, binary_dilation(mask_loc_lh.get_fdata(), mask=mask_lh.get_fdata()))
    n_voxels_overlap_lh = mask_loc_lh.get_fdata().sum()
logging.info(f'{n_voxels_overlap_lh} in overlap_lh.')

n_voxels_overlap_rh = mask_loc_rh.get_fdata().sum()
while n_voxels_overlap_rh < 200:
    logging.info(f'Only {n_voxels_overlap_rh} in overlap_rh. Dilating.')
    mask_loc_rh = new_img_like(mask_loc_rh, binary_dilation(mask_loc_rh.get_fdata(), mask=mask_rh.get_fdata()))
    n_voxels_overlap_rh = mask_loc_rh.get_fdata().sum()
logging.info(f'{n_voxels_overlap_rh} in overlap_rh.')

# Masks
n_voxels_localizer = int(mask_localizer.get_fdata().sum())
n_voxels_lateraloccipital = int(ctx_lateraloccipital.get_fdata().sum())
n_voxels_overlap = int(mask_loc.get_fdata().sum())

f, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
plotting.plot_roi(ctx_lateraloccipital,
                  bg_img=template,
                  black_bg=False,
                  axes=axes[0],
                  title=f'LOC Surface Registration ({subject_id}, {n_voxels_lateraloccipital} voxels)')
plotting.plot_roi(mask_localizer,
                  bg_img=template,
                  black_bg=False,
                  axes=axes[1],
                  title=f'LOC Localizer ({subject_id}, {n_voxels_localizer} voxels)')
plotting.plot_roi(mask_loc,
                  bg_img=template,
                  black_bg=False,
                  axes=axes[2],
                  title=f'LOC Overlap ({subject_id}, {n_voxels_overlap} voxels)')
plotting.plot_stat_map(zstat_localizer,
                       bg_img=template,
                       black_bg=False,
                       axes=axes[3],
                       title=f'Localizer zstat {subject_id}')
f.savefig(f'{output_dir}/images/sub-{subject_id}_masks.pdf')
plt.close(f)

# Argsort zstats to order the voxels in importance
zstat_masked = np.abs(zstat_localizer.get_fdata()[np.where(mask_loc.get_fdata())])
argsort_zstat_masked = np.argsort(zstat_masked)[::-1]
sorted_zstat_masked = zstat_masked[argsort_zstat_masked]

zstat_masked_lh = np.abs(zstat_localizer.get_fdata()[np.where(mask_loc_lh.get_fdata())])
argsort_zstat_masked_lh = np.argsort(zstat_masked_lh)[::-1]
sorted_zstat_masked_lh = zstat_masked_lh[argsort_zstat_masked_lh]

zstat_masked_rh = np.abs(zstat_localizer.get_fdata()[np.where(mask_loc_rh.get_fdata())])
argsort_zstat_masked_rh = np.argsort(zstat_masked_rh)[::-1]
sorted_zstat_masked_rh = zstat_masked_rh[argsort_zstat_masked_rh]

# Show voxel selections
img_10 = math_img(f'(img > {sorted_zstat_masked[10]}) & mask', img=zstat_localizer, mask=mask_loc)
img_50 = math_img(f'(img > {sorted_zstat_masked[50]}) & mask', img=zstat_localizer, mask=mask_loc)
img_100 = math_img(f'(img > {sorted_zstat_masked[100]}) & mask', img=zstat_localizer, mask=mask_loc)
img_200 = math_img(f'(img > {sorted_zstat_masked[200]}) & mask', img=zstat_localizer, mask=mask_loc)
img_400 = math_img(f'(img > {sorted_zstat_masked[400]}) & mask', img=zstat_localizer, mask=mask_loc)
img_800 = math_img(f'(img > {sorted_zstat_masked[800]}) & mask', img=zstat_localizer, mask=mask_loc)

assert img_10.get_fdata().sum() == 10
assert img_50.get_fdata().sum() == 50
assert img_100.get_fdata().sum() == 100
assert img_200.get_fdata().sum() == 200
assert img_400.get_fdata().sum() == 400
assert img_800.get_fdata().sum() == 800

f, ax = plt.subplots(3, 2, figsize=(18, 10))
ax = ax.flatten()
plotting.plot_roi(img_10, bg_img=template, black_bg=False, title=f'Top 10 voxels {subject_id}', axes=ax[0])
plotting.plot_roi(img_50, bg_img=template, black_bg=False, title=f'Top 50 voxels {subject_id}', axes=ax[1])
plotting.plot_roi(img_100, bg_img=template, black_bg=False, title=f'Top 100 voxels {subject_id}', axes=ax[2])
plotting.plot_roi(img_200, bg_img=template, black_bg=False, title=f'Top 200 voxels {subject_id}', axes=ax[3])
plotting.plot_roi(img_400, bg_img=template, black_bg=False, title=f'Top 400 voxels {subject_id}', axes=ax[4])
plotting.plot_roi(img_800, bg_img=template, black_bg=False, title=f'Top 800 voxels {subject_id}', axes=ax[5])
f.savefig(f'{output_dir}/images/sub-{subject_id}_voxel_selection.pdf')
plt.close(f)

# z-scores
f, ax = plt.subplots()
ax.plot(sorted_zstat_masked[:800])
ax.set_title(f'Z-Score Distribution ({subject_id})')
ax.set_xlabel('Voxel')
ax.set_ylabel('Z-Score')
f.savefig(f'{output_dir}/images/sub-{subject_id}_zscore_distribution.pdf')
plt.close(f)

# Skip pe 0, not included
n_excluded_pes = 1
n_pes = 60 - n_excluded_pes


def load_pes_lss(mask, subtract_mean=True, rescale=True, sort_by=None):
    """LS-S regression with one regressor for each event

       Run a separate GLM for each event

    """
    pes = []
    for run in range(1, 9):
        pes_run = []
        for event in range(n_excluded_pes, 60):
            pe = load_img(f'{lss_dir}/sub-{subject_id}/run-{run}/parameter_estimates/fwhm-5.0_event-{event}_pe1.nii.gz')
            # Load the zstat data for the mask
            data = pe.get_fdata()[np.where(mask.get_fdata())]
            if np.isnan(data).any():
                raise ValueError
            pes_run.append(data)
        pes_run = np.vstack(pes_run)
        pes.append(pes_run)
    pes = np.stack(pes)

    if sort_by is not None:
        pes = pes[:, :, sort_by]

    if subtract_mean:
        # Find and subtract the mean of each voxel, for each run
        # Take the mean over events, then re-add a single dimension, and subtract
        pes = pes - np.expand_dims(pes.mean(1), 1)

    # Find and mask unvarying voxels
    nonzero_voxels = (pes.std(1) > 0).all(0)
    pes = pes[:, :, nonzero_voxels]

    if rescale:
        # Rescale each voxel to 1
        pes = pes / np.expand_dims(pes.std(1), 1)

    if np.isnan(pes).any():
        raise ValueError

    # Reshape to single array
    pes = pes.reshape(pes.shape[0] * pes.shape[1], -1)

    return pes


def load_zstats_lss(mask, subtract_mean=True, rescale=True, sort_by=None):
    """LS-S regression with one regressor for each event

       Run a separate GLM for each event

    """
    zstats = []
    for run in range(1, 9):
        zstats_run = []
        for event in range(n_excluded_pes, 60):
            zstat = load_img(f'{lss_dir}/sub-{subject_id}/run-{run}/zfiles/fwhm-5.0_event-{event}_zstat1.nii.gz')
            # Load the zstat data for the mask
            data = zstat.get_fdata()[np.where(mask.get_fdata())]
            if np.isnan(data).any():
                raise ValueError
            zstats_run.append(data)
        zstats_run = np.vstack(zstats_run)
        zstats.append(zstats_run)
    zstats = np.stack(zstats)

    if sort_by is not None:
        zstats = zstats[:, :, sort_by]

    if subtract_mean:
        # Find and subtract the mean of each voxel, for each run
        # Take the mean over events, then re-add a single dimension, and subtract
        zstats = zstats - np.expand_dims(zstats.mean(1), 1)

    # Find and mask unvarying voxels
    nonzero_voxels = (zstats.std(1) > 0).all(0)
    zstats = zstats[:, :, nonzero_voxels]

    if rescale:
        # Rescale each voxel to 1
        zstats = zstats / np.expand_dims(zstats.std(1), 1)

    if np.isnan(zstats).any():
        raise ValueError

    # Reshape to single array
    zstats = zstats.reshape(zstats.shape[0] * zstats.shape[1], -1)

    return zstats


# Nodes and blocks
logging.info('Loading nodes and blocks')

events = []
for run in range(1, 9):
    filename = f'{data_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-{run}_events.tsv'
    data = pd.read_csv(filename, sep='\t').drop(columns=['stim_file', 'trial_type'])
    data['steadystate'] = True
    data.loc[:n_excluded_pes - 1, 'steadystate'] = False
    data['block'] = run - 1
    events.append(data)
events = pd.concat(events).reset_index(drop=True)
events['subject'] = subject_id
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
plt.close(f)

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
plt.close(f)

# Label Clusters
cluster_1 = (events['node'] < 5)
cluster_2 = (events['node'] < 10) & (events['node'] > 4)
cluster_3 = (events['node'] < 15) & (events['node'] > 9)
events['cluster'] = 0
events.loc[cluster_1, 'cluster'] = 1
events.loc[cluster_2, 'cluster'] = 2
events.loc[cluster_3, 'cluster'] = 3
events['cluster_prev'] = events.groupby('block')['cluster'].shift(1).astype('Int8')

# Label shifted clusters
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

# Cross-Cluster
events['cross_cluster'] = False
cross_cluster = (events['cluster'] != events['cluster_prev']) \
    & (events['block'] == events['block_prev'])
events.loc[cross_cluster, 'cross_cluster'] = True

# Border nodes
events['border'] = False
events.loc[events['node'] == 0, 'border'] = True
events.loc[events['node'] == 4, 'border'] = True
events.loc[events['node'] == 5, 'border'] = True
events.loc[events['node'] == 9, 'border'] = True
events.loc[events['node'] == 10, 'border'] = True
events.loc[events['node'] == 14, 'border'] = True


# Label node groups
# Each node within a cluster is 0-4
events['node_group'] = events['node'] % 5

# SVM Setup
n_features_list = (10, 20, 50, 100, 200, 400, 600, 800)
observed = nodes_lss
max_iter = 400000


# Classifiers
def classify_node(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials['node'].astype(int).values
    blocks = trials['block']
    clf = LinearSVC(max_iter=max_iter)
    gkf = LeaveOneGroupOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df['node'] == df['prediction']
    return df
# Only classify trials with correct responses
def classify_node_correct(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']].loc[events['correct']]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials['node'].astype(int).values
    blocks = trials['block']
    clf = LinearSVC(max_iter=max_iter)
    gkf = LeaveOneGroupOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df['node'] == df['prediction']
    return df
def classify_node_hamiltonian(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate'] & events['is_hamiltonian']]
    df = trials.copy()
    if len(trials) > 0:
        trial_inds = trials['pe_ind'].astype(int).values
        x = data[trial_inds, :]
        y = trials['node'].astype(int).values
        blocks = trials['block']
        clf = LinearSVC(max_iter=max_iter)
        gkf = LeaveOneGroupOut()
        with parallel_backend('threading', n_jobs=n_threads):
            predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
        df['prediction'] = predictions
        df['correct_prediction'] = df['node'] == df['prediction']
    return df
def classify_node_randomwalk(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate'] & ~events['is_hamiltonian']]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials['node'].astype(int).values
    blocks = trials['block']
    clf = LinearSVC(max_iter=max_iter)
    gkf = LeaveOneGroupOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df['node'] == df['prediction']
    return df


class LeaveOneGroupVariationOut:
    """
    Custom K-Fold class to exclude the same variation of shapes.

    Each split selects a block and a variation.
    The test set is all shapes of that variation in the target block
    The training set is all shaps of other variations, in all other blocks.
    """

    def __init__(self):
        pass

    def split(self, X, y=None, groups=None):
        trials = groups
        for block in range(8):
            for variation in range(5):
                train_idx = trials.loc[(trials['block'] != block) & (trials['variation'] != variation), 'pe_ind'].astype('int').values
                test_idx = trials.loc[(trials['block'] == block) & (trials['variation'] == variation), 'pe_ind'].astype('int').values
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return 40


def classify_node_loo_variation(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials['node'].astype(int).values
    clf = LinearSVC(max_iter=max_iter)
    logvo = LeaveOneGroupVariationOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=logvo, groups=trials, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df['node'] == df['prediction']
    return df


def classify_cluster(data, events, shift_ind):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials[f'cluster_shifted_{shift_ind}'].astype(int).values
    blocks = trials['block']
    clf = LinearSVC(max_iter=max_iter)
    gkf = LeaveOneGroupOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df[f'cluster_shifted_{shift_ind}'] == df['prediction']
    return df
def classify_cluster_hamiltonian(data, events, shift_ind):
    # Data are the data for those trials
    trials = events.loc[events['steadystate'] & events['is_hamiltonian']]
    df = trials.copy()
    if len(trials) > 0:
        trial_inds = trials['pe_ind'].astype(int).values
        x = data[trial_inds, :]
        y = trials[f'cluster_shifted_{shift_ind}'].astype(int).values
        blocks = trials['block']
        clf = LinearSVC(max_iter=max_iter)
        gkf = LeaveOneGroupOut()
        with parallel_backend('threading', n_jobs=n_threads):
            predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
        df['prediction'] = predictions
        df['correct_prediction'] = df[f'cluster_shifted_{shift_ind}'] == df['prediction']
    return df
def classify_cluster_randomwalk(data, events, shift_ind):
    # Data are the data for those trials
    trials = events.loc[events['steadystate'] & ~events['is_hamiltonian']]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials[f'cluster_shifted_{shift_ind}'].astype(int).values
    blocks = trials['block']
    clf = LinearSVC(max_iter=max_iter)
    gkf = LeaveOneGroupOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df[f'cluster_shifted_{shift_ind}'] == df['prediction']
    return df


def classify_cluster_heldout_nodes(data, events, shift_ind):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials[f'cluster_shifted_{shift_ind}'].astype(int).values
    groups = trials['block'].astype(str) + '_' + trials['node_group'].astype(str)
    clf = LinearSVC(max_iter=max_iter)
    gkf = LeaveOneGroupOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=gkf, groups=groups, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df[f'cluster_shifted_{shift_ind}'] == df['prediction']
    return df


def classify_cross_cluster(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials['cross_cluster'].values
    blocks = trials['block']
    clf = LinearSVC(max_iter=max_iter)
    gkf = LeaveOneGroupOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df['cross_cluster'] == df['prediction']
    return df

def classify_cross_cluster_hamiltonian(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate'] & events['is_hamiltonian']]
    df = trials.copy()
    if len(trials) > 0:
        trial_inds = trials['pe_ind'].astype(int).values
        x = data[trial_inds, :]
        y = trials['cross_cluster'].values
        blocks = trials['block']
        clf = LinearSVC(max_iter=max_iter)
        gkf = LeaveOneGroupOut()
        with parallel_backend('threading', n_jobs=n_threads):
            predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
        df['prediction'] = predictions
        df['correct_prediction'] = df['cross_cluster'] == df['prediction']
    return df

def classify_cross_cluster_randomwalk(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate'] & ~events['is_hamiltonian']]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials['cross_cluster'].values
    blocks = trials['block']
    clf = LinearSVC(max_iter=max_iter)
    gkf = LeaveOneGroupOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df['cross_cluster'] == df['prediction']
    return df

"""
Only classify border nodes
"""
def classify_cross_cluster_border(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']
                        & events['border']]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials['cross_cluster'].values
    blocks = trials['block']
    clf = LinearSVC(max_iter=max_iter)
    gkf = LeaveOneGroupOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df['cross_cluster'] == df['prediction']
    return df

def classify_cross_cluster_border_hamiltonian(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']
                        & events['border']
                        & events['is_hamiltonian']]
    df = trials.copy()
    if len(trials) > 0:
        trial_inds = trials['pe_ind'].astype(int).values
        x = data[trial_inds, :]
        y = trials['cross_cluster'].values
        blocks = trials['block']
        clf = LinearSVC(max_iter=max_iter)
        gkf = LeaveOneGroupOut()
        with parallel_backend('threading', n_jobs=n_threads):
            predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
        df['prediction'] = predictions
        df['correct_prediction'] = df['cross_cluster'] == df['prediction']
    return df

def classify_cross_cluster_border_randomwalk(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']
                        & events['border']
                        & ~events['is_hamiltonian']]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials['cross_cluster'].values
    blocks = trials['block']
    clf = LinearSVC(max_iter=max_iter)
    gkf = LeaveOneGroupOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df['cross_cluster'] == df['prediction']
    return df

"""
Classify nodes within each cluster

Subsetting but hopefully less problematic than past/future node?
"""
def classify_within_each_cluster(data, events):
    df = events.loc[events['steadystate']].copy()
    for cluster in range(1, 4):
        cluster_trials = df[df['cluster'] == cluster]
        trial_inds = cluster_trials['pe_ind'].astype(int).values
        x = data[trial_inds, :]
        y = cluster_trials['node'].astype(int).values
        blocks = cluster_trials['block']
        clf = LinearSVC(max_iter=max_iter)
        gkf = LeaveOneGroupOut()
        with parallel_backend('threading', n_jobs=n_threads):
            preds = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
        df.loc[cluster_trials.index, 'prediction'] = preds
    df['correct_prediction'] = df['node'] == df['prediction']
    return df

def classify_within_each_cluster_hamiltonian(data, events):
    df = events.loc[events['steadystate'] & events['is_hamiltonian']].copy()
    if len(df) > 0:
        for cluster in range(1, 4):
            cluster_trials = df[df['cluster'] == cluster]
            trial_inds = cluster_trials['pe_ind'].astype(int).values
            x = data[trial_inds, :]
            y = cluster_trials['node'].astype(int).values
            blocks = cluster_trials['block']
            clf = LinearSVC(max_iter=max_iter)
            gkf = LeaveOneGroupOut()
            with parallel_backend('threading', n_jobs=n_threads):
                preds = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
            df.loc[cluster_trials.index, 'prediction'] = preds
        df['correct_prediction'] = df['node'] == df['prediction']
    return df

def classify_within_each_cluster_randomwalk(data, events):
    df = events.loc[events['steadystate'] & ~events['is_hamiltonian']].copy()
    for cluster in range(1, 4):
        cluster_trials = df[df['cluster'] == cluster]
        trial_inds = cluster_trials['pe_ind'].astype(int).values
        x = data[trial_inds, :]
        y = cluster_trials['node'].astype(int).values
        blocks = cluster_trials['block']
        clf = LinearSVC(max_iter=max_iter)
        gkf = LeaveOneGroupOut()
        with parallel_backend('threading', n_jobs=n_threads):
            preds = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
        df.loc[cluster_trials.index, 'prediction'] = preds
    df['correct_prediction'] = df['node'] == df['prediction']
    return df


# Previous and Next Nodes
"""
Observe the current functional activity. Can we predict the previous node above chance?
"""
def classify_prev_node_conditionedon_current(data, events):
    df = events.loc[events['steadystate'] & (events['block'] == events['block_prev'])].copy()
    for node in range(15):
        # Trials where the current node is 'node', and the previous node was in the same block
        prev_trials = df[df['node'] == node]
        trial_inds = prev_trials['pe_ind'].astype(int).values
        # Data are the data for those trials
        x = data[trial_inds, :]
        # Observations are the labels for the /previous/ trials
        y = prev_trials['node_prev'].astype(int).values
        blocks = prev_trials['block']
        clf = LinearSVC(max_iter=max_iter)
        gkf = LeaveOneGroupOut()
        with parallel_backend('threading', n_jobs=n_threads):
            preds = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
        df.loc[prev_trials.index, 'prediction'] = preds
    df['correct_prediction'] = df['node_prev'] == df['prediction']
    return df
def classify_prev_node(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']
                        & (events['block'] == events['block_prev'])]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials['node_prev'].astype(int).values
    blocks = trials['block']
    clf = LinearSVC(max_iter=max_iter)
    gkf = LeaveOneGroupOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df['node_prev'] == df['prediction']
    return df
def classify_prev_node_hamiltonian(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']
                        & (events['block'] == events['block_prev'])
                        & events['is_hamiltonian']]
    df = trials.copy()
    if len(trials) > 0:
        trial_inds = trials['pe_ind'].astype(int).values
        x = data[trial_inds, :]
        y = trials['node_prev'].astype(int).values
        blocks = trials['block']
        clf = LinearSVC(max_iter=max_iter)
        gkf = LeaveOneGroupOut()
        with parallel_backend('threading', n_jobs=n_threads):
            predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
        df['prediction'] = predictions
        df['correct_prediction'] = df['node_prev'] == df['prediction']
    return df
def classify_prev_node_randomwalk(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']
                        & (events['block'] == events['block_prev'])
                        & ~events['is_hamiltonian']]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials['node_prev'].astype(int).values
    blocks = trials['block']
    clf = LinearSVC(max_iter=max_iter)
    gkf = LeaveOneGroupOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df['node_prev'] == df['prediction']
    return df

"""
Observe the current functional activity. Can we predict the next node above chance?

Excluding Hamiltonian biases, this really shouldn't be possible above 1/4
"""
def classify_next_node_conditionedon_current(data, events):
    df = events.loc[events['steadystate']
                & (events['block'] == events['block_next'])].copy()
    for node in range(15):
        # Trials where the current node is 'node', and the next node is in the same block
        # We also ensure the previous block was the same, to not use the first PE
        next_trials = df[df['node'] == node]
        trial_inds = next_trials['pe_ind'].astype(int).values
        # Data are the data for those trials
        x = data[trial_inds, :]
        # Observations are the labels for the /next/ trials
        y = next_trials['node_next'].astype(int).values
        blocks = next_trials['block']
        clf = LinearSVC(max_iter=max_iter)
        gkf = LeaveOneGroupOut()
        with parallel_backend('threading', n_jobs=n_threads):
            preds = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
        df.loc[next_trials.index, 'prediction'] = preds
    df['correct_prediction'] = df['node_next'] == df['prediction']
    return df
def classify_next_node(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']
                        & (events['block'] == events['block_next'])]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials['node_next'].astype(int).values
    blocks = trials['block']
    clf = LinearSVC(max_iter=max_iter)
    gkf = LeaveOneGroupOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df['node_next'] == df['prediction']
    return df
def classify_next_node_hamiltonian(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']
                        & (events['block'] == events['block_next'])
                        & events['is_hamiltonian']]
    df = trials.copy()
    if len(trials) > 0:
        trial_inds = trials['pe_ind'].astype(int).values
        x = data[trial_inds, :]
        y = trials['node_next'].astype(int).values
        blocks = trials['block']
        clf = LinearSVC(max_iter=max_iter)
        gkf = LeaveOneGroupOut()
        with parallel_backend('threading', n_jobs=n_threads):
            predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
        df['prediction'] = predictions
        df['correct_prediction'] = df['node_next'] == df['prediction']
    return df
def classify_next_node_randomwalk(data, events):
    # Data are the data for those trials
    trials = events.loc[events['steadystate']
                        & (events['block'] == events['block_next'])
                        & ~events['is_hamiltonian']]
    trial_inds = trials['pe_ind'].astype(int).values
    x = data[trial_inds, :]
    y = trials['node_next'].astype(int).values
    blocks = trials['block']
    clf = LinearSVC(max_iter=max_iter)
    gkf = LeaveOneGroupOut()
    with parallel_backend('threading', n_jobs=n_threads):
        predictions = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
    df = trials.copy()
    df['prediction'] = predictions
    df['correct_prediction'] = df['node_next'] == df['prediction']
    return df

"""
If we take the subset of nodes where the next node was 'x',
can we classify the current node above chance?

It may be the case that if the next node is node 5,
consistently guessing node 4 gives above-chance performance

If, for example, due to the Hamiltonian walks, the transition 4->5 is more common than
4->anything else
"""
def classify_current_conditionedon_next(data, events):
    df = events.loc[events['steadystate']
                & (events['block'] == events['block_next'])
                & (events['block'] == events['block_prev'])].copy()
    for next_node in range(15):
        trials = df[df['node_next'] == next_node]
        # Data are the data for those trials
        trial_inds = trials['pe_ind'].astype(int)
        x = data[trial_inds, :]
        # Observations are the labels for the /previous/ trials
        y = trials['node'].astype(int).values
        blocks = trials['block']
        clf = LinearSVC(max_iter=max_iter)
        gkf = LeaveOneGroupOut()
        with parallel_backend('threading', n_jobs=n_threads):
            preds = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
        df.loc[trials.index, 'prediction'] = preds
    df['correct_prediction'] = df['node'] == df['prediction']
    return df
def classify_current_conditionedon_next_hamiltonian(data, events):
    df = events.loc[events['steadystate']
                & events['is_hamiltonian']
                & (events['block'] == events['block_next'])
                & (events['block'] == events['block_prev'])].copy()
    if len(df) > 0:
        for next_node in range(15):
            trials = df[df['node_next'] == next_node]
            # Data are the data for those trials
            trial_inds = trials['pe_ind'].astype(int)
            x = data[trial_inds, :]
            # Observations are the labels for the /previous/ trials
            y = trials['node'].astype(int).values
            blocks = trials['block']
            clf = LinearSVC(max_iter=max_iter)
            gkf = LeaveOneGroupOut()
            with parallel_backend('threading', n_jobs=n_threads):
                preds = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
            df.loc[trials.index, 'prediction'] = preds
        df['correct_prediction'] = df['node'] == df['prediction']
    return df
def classify_current_conditionedon_next_randomwalk(data, events):
    df = events.loc[events['steadystate']
                & ~events['is_hamiltonian']
                & (events['block'] == events['block_next'])
                & (events['block'] == events['block_prev'])].copy()
    for next_node in range(15):
        trials = df[df['node_next'] == next_node]
        # Data are the data for those trials
        trial_inds = trials['pe_ind'].astype(int)
        x = data[trial_inds, :]
        # Observations are the labels for the /previous/ trials
        y = trials['node'].astype(int).values
        blocks = trials['block']
        clf = LinearSVC(max_iter=max_iter)
        gkf = LeaveOneGroupOut()
        with parallel_backend('threading', n_jobs=n_threads):
            preds = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
        df.loc[trials.index, 'prediction'] = preds
    df['correct_prediction'] = df['node'] == df['prediction']
    return df

def classify_current_conditionedon_prev(data, events):
    df = events.loc[events['steadystate'] & (events['block'] == events['block_prev'])].copy()
    for prev_node in range(15):
        # Trials where the current node is 'node', and the previous node was in the same block
        trials = df[df['node_prev'] == prev_node]
        # Data are the data for those trials
        trial_inds = trials['pe_ind'].astype(int)
        x = data[trial_inds, :]
        # Observations are the labels for the /previous/ trials
        y = trials['node'].astype(int).values
        blocks = trials['block']
        clf = LinearSVC(max_iter=max_iter)
        gkf = LeaveOneGroupOut()
        with parallel_backend('threading', n_jobs=n_threads):
            preds = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
        df.loc[trials.index, 'prediction'] = preds
    df['correct_prediction'] = df['node'] == df['prediction']
    return df
def classify_current_conditionedon_prev_hamiltonian(data, events):
    df = events.loc[events['steadystate'] & events['is_hamiltonian'] & (events['block'] == events['block_prev'])].copy()
    if len(df) > 0:
        for prev_node in range(15):
            # Trials where the current node is 'node', and the previous node was in the same block
            trials = df[df['node_prev'] == prev_node]
            # Data are the data for those trials
            trial_inds = trials['pe_ind'].astype(int)
            x = data[trial_inds, :]
            # Observations are the labels for the /previous/ trials
            y = trials['node'].astype(int).values
            blocks = trials['block']
            clf = LinearSVC(max_iter=max_iter)
            gkf = LeaveOneGroupOut()
            with parallel_backend('threading', n_jobs=n_threads):
                preds = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
            df.loc[trials.index, 'prediction'] = preds
        df['correct_prediction'] = df['node'] == df['prediction']
    return df
def classify_current_conditionedon_prev_randomwalk(data, events):
    df = events.loc[events['steadystate'] & ~events['is_hamiltonian'] & (events['block'] == events['block_prev'])].copy()
    for prev_node in range(15):
        # Trials where the current node is 'node', and the previous node was in the same block
        trials = df[df['node_prev'] == prev_node]
        # Data are the data for those trials
        trial_inds = trials['pe_ind'].astype(int)
        x = data[trial_inds, :]
        # Observations are the labels for the /previous/ trials
        y = trials['node'].astype(int).values
        blocks = trials['block']
        clf = LinearSVC(max_iter=max_iter)
        gkf = LeaveOneGroupOut()
        with parallel_backend('threading', n_jobs=n_threads):
            preds = cross_val_predict(clf, x, y, cv=gkf, groups=blocks, verbose=False)
        df.loc[trials.index, 'prediction'] = preds
    df['correct_prediction'] = df['node'] == df['prediction']
    return df


classifiers = [
    # ('cluster-shifted-1', partial(classify_cluster, shift_ind=1)),
    # ('cluster-shifted-2', partial(classify_cluster, shift_ind=2)),
    # ('cluster-shifted-3', partial(classify_cluster, shift_ind=3)),
    # ('cluster-shifted-4', partial(classify_cluster, shift_ind=4)),
    # ('cluster-heldout-shifted-0', partial(classify_cluster_heldout_nodes, shift_ind=0)),
    # ('cluster-heldout-shifted-1', partial(classify_cluster_heldout_nodes, shift_ind=1)),
    # ('cluster-heldout-shifted-2', partial(classify_cluster_heldout_nodes, shift_ind=2)),
    # ('cluster-heldout-shifted-3', partial(classify_cluster_heldout_nodes, shift_ind=3)),
    # ('cluster-heldout-shifted-4', partial(classify_cluster_heldout_nodes, shift_ind=4)),
    # ('node-loo-variation', classify_node_loo_variation),
    # ('next-node', classify_next_node),
    # ('next-node-hamiltonian', classify_next_node_hamiltonian),
    # ('next-node-randomwalk', classify_next_node_randomwalk),
    # ('prev-node', classify_prev_node),
    # ('prev-node-hamiltonian', classify_prev_node_hamiltonian),
    # ('prev-node-randomwalk', classify_prev_node_randomwalk),
    # ('next-node-conditioned-current', classify_next_node_conditionedon_current),
    # ('prev-node-conditioned-current', classify_prev_node_conditionedon_current),
    # ('current-from-prev', classify_current_conditionedon_prev),
    # ('current-from-prev-hamiltonian', classify_current_conditionedon_prev_hamiltonian),
    # ('current-from-prev-randomwalk', classify_current_conditionedon_prev_randomwalk),
    # ('current-from-next', classify_current_conditionedon_next),
    # ('current-from-next-hamiltonian', classify_current_conditionedon_next_hamiltonian),
    # ('current-from-next-randomwalk', classify_current_conditionedon_next_randomwalk),

    # ('cluster-shifted-0', partial(classify_cluster, shift_ind=0)),
    ('node', classify_node),
    ('node_correct', classify_node_correct),
    # ('cross-cluster', classify_cross_cluster),
    # ('cross-cluster-border', classify_cross_cluster_border),
    # ('node-withincluster', classify_within_each_cluster),

    # ('cluster-shifted-0-randomwalk', partial(classify_cluster_randomwalk, shift_ind=0)),
    # ('node-randomwalk', classify_node_randomwalk),
    # ('cross-cluster-randomwalk', classify_cross_cluster_randomwalk),
    # ('cross-cluster-border-randomwalk', classify_cross_cluster_border_randomwalk),
    # ('node-withincluster-randomwalk', classify_within_each_cluster_randomwalk),

    # ('cluster-shifted-0-hamiltonian', partial(classify_cluster_hamiltonian, shift_ind=0)),
    # ('node-hamiltonian', classify_node_hamiltonian),
    # ('cross-cluster-hamiltonian', classify_cross_cluster_hamiltonian),
    # ('cross-cluster-border-hamiltonian', classify_cross_cluster_border_hamiltonian),
    # ('node-withincluster-hamiltonian', classify_within_each_cluster_hamiltonian),
]

pe_params = (
    (True, True),
    (False, True),
    (True, False),
    (False, False),
)


rois = (
    ('hippocampus-both', partial(load_hippocampus, hemi='both')),
    ('hippocampus-lh', partial(load_hippocampus, hemi='left')),
    ('hippocampus-rh', partial(load_hippocampus, hemi='right')),
    ('entorhinal-both', partial(load_entorhinal, hemi='both')),
    ('entorhinal-lh', partial(load_entorhinal, hemi='left')),
    ('entorhinal-rh', partial(load_entorhinal, hemi='right')),
    #('lingual-both', partial(load_lingual, hemi='both')),
    #('lingual-lh', partial(load_lingual, hemi='left')),
    #('lingual-rh', partial(load_lingual, hemi='right')),
    #('pericalcarine-both', partial(load_pericalcarine, hemi='both')),
    #('pericalcarine-lh', partial(load_pericalcarine, hemi='left')),
    #('pericalcarine-rh', partial(load_pericalcarine, hemi='right')),
    #('cuneus-both', partial(load_cuneus, hemi='both')),
    #('cuneus-lh', partial(load_cuneus, hemi='left')),
    #('cuneus-rh', partial(load_cuneus, hemi='right')),
    ('lateraloccipital-both', partial(load_loc, hemi='both')),
    ('lateraloccipital-lh', partial(load_loc, hemi='left')),
    ('lateraloccipital-rh', partial(load_loc, hemi='right')),
    ('postcentral-both', partial(load_postcentral, hemi='both')),
    ('postcentral-lh', partial(load_postcentral, hemi='left')),
    ('postcentral-rh', partial(load_postcentral, hemi='right')),
    ('precentral-both', partial(load_precentral, hemi='both')),
    ('precentral-lh', partial(load_precentral, hemi='left')),
    ('precentral-rh', partial(load_precentral, hemi='right')),
    # ('benson-1', partial(load_benson_roi, roi=1)),
    # ('benson-2', partial(load_benson_roi, roi=2)),
    # ('benson-3', partial(load_benson_roi, roi=3)),
    # ('benson-4', partial(load_benson_roi, roi=4)),
    # ('benson-5', partial(load_benson_roi, roi=5)),
    # ('benson-6', partial(load_benson_roi, roi=6)),
    # ('benson-7', partial(load_benson_roi, roi=7)),
    # ('benson-8', partial(load_benson_roi, roi=8)),
    # ('benson-9', partial(load_benson_roi, roi=9)),
    # ('benson-10', partial(load_benson_roi, roi=10)),
    # ('benson-11', partial(load_benson_roi, roi=11)),
    # ('benson-12', partial(load_benson_roi, roi=12)),
)

for (roi_name, roi_mask_func) in rois:
    logging.info(f'Loading mask {roi_name}...')
    mask_roi = roi_mask_func(subject_id=subject_id)
    n_voxels_roi = int(mask_roi.get_fdata().sum())

    f, ax = plt.subplots()
    plotting.plot_roi(mask_roi,
                      bg_img=template,
                      black_bg=False,
                      axes=ax,
                      title=f'{roi_name} / {subject_id} / {n_voxels_roi} voxels)')
    f.savefig(f'{output_dir}/images/sub-{subject_id}_{roi_name}_mask.pdf')
    plt.close(f)

    for subtract_mean, rescale in pe_params:
        logging.info(f'Condition: subtract_mean={subtract_mean}, rescale={rescale}')
        logging.info(f'Loading zstats {roi_name} LS-S...')
        zstats_lss = load_zstats_lss(mask_roi, subtract_mean=subtract_mean, rescale=rescale)
        logging.info(f'{zstats_lss.shape[0]} events')
        logging.info(f'{zstats_lss.shape[1]} voxels')

        f, ax = plt.subplots(ncols=2, figsize=(10, 4))
        ax[0].plot(zstats_lss[:, 0])
        ax[0].set_title('Example Voxel')
        ax[0].set_xlabel('Trial')
        ax[0].set_ylabel('Amplitude')

        ax[1].plot(zstats_lss[0, :])
        ax[1].set_title('Example Trial')
        ax[1].set_xlabel('Voxel')
        ax[1].set_ylabel('Amplitude')
        f.suptitle(f'{roi_name} / {subject_id}')
        f.tight_layout()
        f.savefig(f'{output_dir}/images/sub-{subject_id}_rescale-{rescale}_subtract-mean-{subtract_mean}_zstats_{roi_name}_sample-zstats.pdf')
        plt.close(f)

        for (classifier_name, classifier) in classifiers:
            logging.info(f'Predicting {classifier_name}: {roi_name}')
            results = classifier(zstats_lss, events)
            results['roi'] = roi_name
            results['classifier'] = classifier_name
            results['subtract_mean'] = subtract_mean
            results['rescale'] = rescale
            results.to_csv(f'{output_dir}/sub-{subject_id}_rescale-{rescale}_subtract-mean-{subtract_mean}_zstats_{roi_name}_predictions-{classifier_name}.csv.gz')
            gc.collect()

# LOC Localizer is a special case
roi_name = 'loc-localized'
logging.info(f'{roi_name}...')
for subtract_mean, rescale in pe_params:
    logging.info(f'Condition: subtract_mean={subtract_mean}, rescale={rescale}')
    logging.info('Loading PEs {roi_name} LS-S...')
    zstats_loc_lss = load_zstats_lss(mask_loc, sort_by=argsort_zstat_masked, subtract_mean=subtract_mean, rescale=rescale)
    logging.info(f'{zstats_loc_lss.shape[0]} events')
    logging.info(f'{zstats_loc_lss.shape[1]} voxels')

    f, ax = plt.subplots(ncols=2, figsize=(10, 4))
    ax[0].plot(zstats_loc_lss[:, 0])
    ax[0].set_title('Example Voxel')
    ax[0].set_xlabel('Trial')
    ax[0].set_ylabel('Amplitude')

    ax[1].plot(zstats_loc_lss[0, :])
    ax[1].set_title('Example Trial')
    ax[1].set_xlabel('Voxel')
    ax[1].set_ylabel('Amplitude')
    f.suptitle(f'{roi_name} / {subject_id}')
    f.tight_layout()
    f.savefig(f'{output_dir}/images/sub-{subject_id}_rescale-{rescale}_subtract-mean-{subtract_mean}_zstats_{roi_name}_sample-zstats.pdf')
    plt.close(f)

    for (classifier_name, classifier) in classifiers:
        logging.info(f'Predicting {classifier_name}')

        logging.info(f'Predicting {classifier_name}: LOC Localized')
        loc_results = pd.DataFrame()
        for i, n_features in enumerate(n_features_list):
            logging.info(f'Predicting {classifier_name}: LOC: {n_features} voxels')
            # Select first `n_features` points
            data = zstats_loc_lss[:, :n_features]
            df = classifier(data, events)
            df['voxels'] = n_features
            loc_results = pd.concat([loc_results, df])
        loc_results = loc_results.reset_index(drop=True)
        loc_results['roi'] = 'loc-localized'
        loc_results['classifier'] = classifier_name
        loc_results['subtract_mean'] = subtract_mean
        loc_results['rescale'] = rescale
        loc_results.to_csv(f'{output_dir}/sub-{subject_id}_rescale-{rescale}_subtract-mean-{subtract_mean}_zstats_{roi_name}_predictions-{classifier_name}.csv.gz')
        gc.collect()

roi_name = 'loc-localized-lh'
logging.info(f'{roi_name}...')
for subtract_mean, rescale in pe_params:
    logging.info(f'Condition: subtract_mean={subtract_mean}, rescale={rescale}')
    logging.info('Loading PEs {roi_name} LS-S...')
    zstats_loc_lss_lh = load_zstats_lss(mask_loc_lh, sort_by=argsort_zstat_masked_lh, subtract_mean=subtract_mean, rescale=rescale)
    logging.info(f'{zstats_loc_lss_lh.shape[0]} events')
    logging.info(f'{zstats_loc_lss_lh.shape[1]} voxels')

    f, ax = plt.subplots(ncols=2, figsize=(10, 4))
    ax[0].plot(zstats_loc_lss_lh[:, 0])
    ax[0].set_title('Example Voxel')
    ax[0].set_xlabel('Trial')
    ax[0].set_ylabel('Amplitude')

    ax[1].plot(zstats_loc_lss_lh[0, :])
    ax[1].set_title('Example Trial')
    ax[1].set_xlabel('Voxel')
    ax[1].set_ylabel('Amplitude')
    f.suptitle(f'{roi_name} / {subject_id}')
    f.tight_layout()
    f.savefig(f'{output_dir}/images/sub-{subject_id}_rescale-{rescale}_subtract-mean-{subtract_mean}_zstats_{roi_name}_sample-zstats.pdf')
    plt.close(f)

    for (classifier_name, classifier) in classifiers:
        logging.info(f'Predicting {classifier_name}')

        logging.info(f'Predicting {classifier_name}: LOC Localized')
        loc_results = pd.DataFrame()
        for i, n_features in enumerate(n_features_list):
            logging.info(f'Predicting {classifier_name}: LOC: {n_features} voxels')
            # Select first `n_features` points
            data = zstats_loc_lss_lh[:, :n_features]
            df = classifier(data, events)
            df['voxels'] = n_features
            loc_results = pd.concat([loc_results, df])
        loc_results = loc_results.reset_index(drop=True)
        loc_results['roi'] = 'loc-localized-lh'
        loc_results['classifier'] = classifier_name
        loc_results['subtract_mean'] = subtract_mean
        loc_results['rescale'] = rescale
        loc_results.to_csv(f'{output_dir}/sub-{subject_id}_rescale-{rescale}_subtract-mean-{subtract_mean}_zstats_{roi_name}_predictions-{classifier_name}.csv.gz')
        gc.collect()

roi_name = 'loc-localized-rh'
logging.info(f'{roi_name}...')
for subtract_mean, rescale in pe_params:
    logging.info(f'Condition: subtract_mean={subtract_mean}, rescale={rescale}')
    logging.info('Loading PEs {roi_name} LS-S...')
    zstats_loc_lss_rh = load_zstats_lss(mask_loc_rh, sort_by=argsort_zstat_masked_rh, subtract_mean=subtract_mean, rescale=rescale)
    logging.info(f'{zstats_loc_lss_rh.shape[0]} events')
    logging.info(f'{zstats_loc_lss_rh.shape[1]} voxels')

    f, ax = plt.subplots(ncols=2, figsize=(10, 4))
    ax[0].plot(zstats_loc_lss_rh[:, 0])
    ax[0].set_title('Example Voxel')
    ax[0].set_xlabel('Trial')
    ax[0].set_ylabel('Amplitude')

    ax[1].plot(zstats_loc_lss_rh[0, :])
    ax[1].set_title('Example Trial')
    ax[1].set_xlabel('Voxel')
    ax[1].set_ylabel('Amplitude')
    f.suptitle(f'{roi_name} / {subject_id}')
    f.tight_layout()
    f.savefig(f'{output_dir}/images/sub-{subject_id}_rescale-{rescale}_subtract-mean-{subtract_mean}_zstats_{roi_name}_sample-zstats.pdf')
    plt.close(f)

    for (classifier_name, classifier) in classifiers:
        logging.info(f'Predicting {classifier_name}')

        logging.info(f'Predicting {classifier_name}: LOC Localized')
        loc_results = pd.DataFrame()
        for i, n_features in enumerate(n_features_list):
            logging.info(f'Predicting {classifier_name}: LOC: {n_features} voxels')
            # Select first `n_features` points
            data = zstats_loc_lss_rh[:, :n_features]
            df = classifier(data, events)
            df['voxels'] = n_features
            loc_results = pd.concat([loc_results, df])
        loc_results = loc_results.reset_index(drop=True)
        loc_results['roi'] = 'loc-localized-rh'
        loc_results['classifier'] = classifier_name
        loc_results['subtract_mean'] = subtract_mean
        loc_results['rescale'] = rescale
        loc_results.to_csv(f'{output_dir}/sub-{subject_id}_rescale-{rescale}_subtract-mean-{subtract_mean}_zstats_{roi_name}_predictions-{classifier_name}.csv.gz')
        gc.collect()
