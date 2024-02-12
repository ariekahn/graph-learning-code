import logging
import pandas as pd
import numpy as np
import os
import argparse
import gc
from nilearn.image import load_img, math_img, new_img_like
from nilearn import plotting
from functools import partial, lru_cache
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('Run MVPA analysis')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('project_dir', type=str, help='directory containing data and derived subfolders')
parser.add_argument('output_dir', type=str, help='Location for output files')
args = parser.parse_args()

subject_id = args.subject_id
project_dir = args.project_dir
output_dir = args.output_dir

logging.basicConfig(filename=f'{output_dir}/mvpa_{subject_id}.log', filemode='w', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info(f'Subject: {subject_id}')

os.makedirs(f'{output_dir}/images', exist_ok=True)

# Configure what subject we're using
space = 'MNI152NLin2009cAsym'

fmriprep_dir = f'{project_dir}/derived/fmriprep'
lss_dir = f'{project_dir}/derived/feat_representation_lss'
localizer_dir = f'{project_dir}/derived/feat_localizer'
data_dir = f'{project_dir}/data'

# Create some helper functions for
# 1. Loading the T1 template
# 2. Loading the LOC template


@lru_cache()
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


@lru_cache()
def load_aparc(subject_id):
    """
    Load Freesurfer aparc for localizer

    derived/fmriprep/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz
    """
    aparc = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz')
    return aparc


@lru_cache()
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


@lru_cache()
def load_benson_parc(subject_id):
    """
    Load Benson varea
    """
    varea = load_img(f'{project_dir}/derived/template_benson/benson14_varea/sub-{subject_id}/sub-{subject_id}_ses-2_task-graphrepresentation_run-1_space-MNI152NLin2009cAsym_benson14_varea.nii.gz')
    return varea


@lru_cache()
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


@lru_cache()
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


@lru_cache()
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
    loc = math_img(roi_str, img=aparc)
    return loc


@lru_cache()
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
    loc = math_img(roi_str, img=aparc)
    return loc


def load_lingual(subject_id, hemi='both'):
    """
    Load a mask for lingual gyrus

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-??? = 1
    ctx-rh-??? = 2
    """
    roi_str = ''
    if hemi == 'left':
        roi_str = 'img == 1013'
    elif hemi == 'right':
        roi_str = 'img == 2013'
    elif hemi == 'both':
        roi_str = '(img == 1013) | (img == 2013)'
    elif hemi == 'separate':
        roi_str = '(img == 1013) + 2*(img == 2013)'
    else:
        raise ValueError('hemi must be left/right/both/separate')

    aparc = load_aparc(subject_id)
    lingual = math_img(roi_str, img=aparc)
    return lingual


def load_pericalcarine(subject_id, hemi='both'):
    """
    Load a mask for pericalcarine gyrus

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-??? = 1
    ctx-rh-??? = 2
    """
    roi_str = ''
    if hemi == 'left':
        roi_str = 'img == 1021'
    elif hemi == 'right':
        roi_str = 'img == 2021'
    elif hemi == 'both':
        roi_str = '(img == 1021) | (img == 2021)'
    elif hemi == 'separate':
        roi_str = '(img == 1021) + 2*(img == 2021)'
    else:
        raise ValueError('hemi must be left/right/both/separate')

    aparc = load_aparc(subject_id)
    pericalcarine = math_img(roi_str, img=aparc)
    return pericalcarine


def load_cuneus(subject_id, hemi='both'):
    """
    Load a mask for cuneus gyrus

    subject_id : str
        GLSxxx
    hemi : str
        'left', 'right', 'both', or 'separate'
    returns:
        nibabel.nifti1.Nifti1Image

    If separate,
    ctx-lh-??? = 1
    ctx-rh-??? = 2
    """
    roi_str = ''
    if hemi == 'left':
        roi_str = 'img == 1005'
    elif hemi == 'right':
        roi_str = 'img == 2005'
    elif hemi == 'both':
        roi_str = '(img == 1005) | (img == 2005)'
    elif hemi == 'separate':
        roi_str = '(img == 1005) + 2*(img == 2005)'
    else:
        raise ValueError('hemi must be left/right/both/separate')

    aparc = load_aparc(subject_id)
    cuneus = math_img(roi_str, img=aparc)
    return cuneus


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
    ctx-lh-??? = 1
    ctx-rh-??? = 2
    """
    roi_str = ''
    if hemi == 'left':
        roi_str = 'img == 1022'
    elif hemi == 'right':
        roi_str = 'img == 2022'
    elif hemi == 'both':
        roi_str = '(img == 1022) | (img == 2022)'
    elif hemi == 'separate':
        roi_str = '(img == 1022) + 2*(img == 2022)'
    else:
        raise ValueError('hemi must be left/right/both/separate')

    aparc = load_aparc(subject_id)
    postcentral = math_img(roi_str, img=aparc)
    return postcentral


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
    ctx-lh-??? = 1
    ctx-rh-??? = 2
    """
    roi_str = ''
    if hemi == 'left':
        roi_str = 'img == 1024'
    elif hemi == 'right':
        roi_str = 'img == 2024'
    elif hemi == 'both':
        roi_str = '(img == 1024) | (img == 2024)'
    elif hemi == 'separate':
        roi_str = '(img == 1024) + 2*(img == 2024)'
    else:
        raise ValueError('hemi must be left/right/both/separate')

    aparc = load_aparc(subject_id)
    precentral = math_img(roi_str, img=aparc)
    return precentral


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


@lru_cache()
def load_bold_mask(subject_id):
    """
    Load bold mask intersection across all runs

    derived/fmriprep/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
    """
    boldmask = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
    return boldmask


os.makedirs(output_dir, exist_ok=True)

# Load T1w template
template = load_masked_template(subject_id)

# Localizer Mask
bold_mask = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-{space}_desc-brain_mask.nii.gz')
bold_mask = new_img_like(bold_mask, np.rint(bold_mask.get_fdata()) > 0)

ctx_lateraloccipital = load_loc(subject_id)

zstat_localizer = load_zstat(subject_id)

# Load cluster assignments
cluster_img = load_zstat_index(subject_id)
# Find the two largest clusters
clusters = np.asarray(cluster_img.get_fdata())
n_clusters = np.max(clusters)
mask_localizer = math_img(f'(img == {n_clusters}) | (img == {n_clusters - 1})', img=cluster_img)
if subject_id in ('GLS024'):
    mask_localizer = math_img(f'(img == {n_clusters}) | (img == {n_clusters - 2})', img=cluster_img)
elif subject_id in ('GLS017'):
    mask_localizer = math_img(f'(img == {n_clusters - 1}) | (img == {n_clusters - 2})', img=cluster_img)

# Combine the localizer and LOC masks
mask_lh = math_img('img > 0', img=load_aseg(subject_id, 'lh'))
mask_rh = math_img('img > 0', img=load_aseg(subject_id, 'rh'))

mask_loc = math_img('img1 & img2', img1=mask_localizer, img2=ctx_lateraloccipital)
mask_loc_lh = math_img('img1 & img2 & img3', img1=mask_localizer, img2=ctx_lateraloccipital, img3=mask_lh)
mask_loc_rh = math_img('img1 & img2 & img3', img1=mask_localizer, img2=ctx_lateraloccipital, img3=mask_rh)

n_voxels_overlap = mask_loc.get_fdata().sum()
while n_voxels_overlap < 800:
    logging.info(f'Only {n_voxels_overlap} in overlap. Dilating.')
    mask_loc = new_img_like(mask_loc, binary_dilation(mask_loc.get_fdata(), mask=bold_mask.get_fdata()))
    n_voxels_overlap = mask_loc.get_fdata().sum()
logging.info(f'{n_voxels_overlap} in overlap.')

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

# Argsort zstats to order the voxels in importance
zstat_masked = zstat_localizer.get_fdata()[np.where(mask_loc.get_fdata())]
argsort_zstat_masked = np.argsort(zstat_masked)[::-1]
sorted_zstat_masked = zstat_masked[argsort_zstat_masked]

zstat_masked_lh = np.abs(zstat_localizer.get_fdata()[np.where(mask_loc_lh.get_fdata())])
argsort_zstat_masked_lh = np.argsort(zstat_masked_lh)[::-1]
sorted_zstat_masked_lh = zstat_masked_lh[argsort_zstat_masked_lh]

zstat_masked_rh = np.abs(zstat_localizer.get_fdata()[np.where(mask_loc_rh.get_fdata())])
argsort_zstat_masked_rh = np.argsort(zstat_masked_rh)[::-1]
sorted_zstat_masked_rh = zstat_masked_rh[argsort_zstat_masked_rh]

img_10 = math_img(f'(img > {sorted_zstat_masked[10]}) & mask', img=zstat_localizer, mask=mask_loc)
img_50 = math_img(f'(img > {sorted_zstat_masked[50]}) & mask', img=zstat_localizer, mask=mask_loc)
img_100 = math_img(f'(img > {sorted_zstat_masked[100]}) & mask', img=zstat_localizer, mask=mask_loc)
img_200 = math_img(f'(img > {sorted_zstat_masked[200]}) & mask', img=zstat_localizer, mask=mask_loc)
img_400 = math_img(f'(img > {sorted_zstat_masked[400]}) & mask', img=zstat_localizer, mask=mask_loc)
img_800 = math_img(f'(img > {sorted_zstat_masked[800]}) & mask', img=zstat_localizer, mask=mask_loc)

n_excluded_pes = 1
n_pes = 60 - n_excluded_pes


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

zstat_params = (
    (True, True),
    (True, False),
    (False, True),
    (False, False)
)

for (rescale, subtract_mean) in zstat_params:
    roi_name = 'loc-localized-both'
    zstats_loc_lss = load_zstats_lss(mask_loc, sort_by=argsort_zstat_masked, rescale=rescale, subtract_mean=subtract_mean)
    logging.info('pes LOC LS-S:')
    logging.info(f'{zstats_loc_lss.shape[0]} events')
    logging.info(f'{zstats_loc_lss.shape[1]} voxels')

    nvoxels = 800
    block_avg_patterns_loc = np.zeros((8, 15, nvoxels))
    for block, node in np.ndindex(8, 15):
        block_avg_patterns_loc[block, node, :] = zstats_loc_lss[(nodes_lss == node) & (blocks_lss == block), :nvoxels].mean(0)
    block_avg_results_loc = np.zeros((15, 15))
    for node_i, node_j, block_i, block_j in np.ndindex(15, 15, 8, 8):
        if block_i != block_j:
            block_avg_results_loc[node_i, node_j] += np.corrcoef(block_avg_patterns_loc[block_i, node_i, :],
                                                                 block_avg_patterns_loc[block_j, node_j, :])[0][1]
    block_avg_results_loc /= 8 * 7
    np.save(f'{output_dir}/sub-{subject_id}_rescale-{rescale}_subtract-mean-{subtract_mean}_cross-val-corr_{roi_name}_zstats_{nvoxels}_block-avg.npy', block_avg_results_loc, allow_pickle=False)

    roi_name = 'loc-localized-lh'
    zstats_loc_lss = load_zstats_lss(mask_loc_lh, sort_by=argsort_zstat_masked_lh, rescale=rescale, subtract_mean=subtract_mean)
    logging.info('pes LOC LS-S:')
    logging.info(f'{zstats_loc_lss.shape[0]} events')
    logging.info(f'{zstats_loc_lss.shape[1]} voxels')

    nvoxels = 200
    block_avg_patterns_loc = np.zeros((8, 15, nvoxels))
    for block, node in np.ndindex(8, 15):
        block_avg_patterns_loc[block, node, :] = zstats_loc_lss[(nodes_lss == node) & (blocks_lss == block), :nvoxels].mean(0)
    block_avg_results_loc = np.zeros((15, 15))
    for node_i, node_j, block_i, block_j in np.ndindex(15, 15, 8, 8):
        if block_i != block_j:
            block_avg_results_loc[node_i, node_j] += np.corrcoef(block_avg_patterns_loc[block_i, node_i, :],
                                                                 block_avg_patterns_loc[block_j, node_j, :])[0][1]
    block_avg_results_loc /= 8 * 7
    np.save(f'{output_dir}/sub-{subject_id}_rescale-{rescale}_subtract-mean-{subtract_mean}_cross-val-corr_{roi_name}_zstats_{nvoxels}_block-avg.npy', block_avg_results_loc, allow_pickle=False)

    roi_name = 'loc-localized-rh'
    zstats_loc_lss = load_zstats_lss(mask_loc_rh, sort_by=argsort_zstat_masked_rh, rescale=rescale, subtract_mean=subtract_mean)
    logging.info('pes LOC LS-S:')
    logging.info(f'{zstats_loc_lss.shape[0]} events')
    logging.info(f'{zstats_loc_lss.shape[1]} voxels')

    nvoxels = 200
    block_avg_patterns_loc = np.zeros((8, 15, nvoxels))
    for block, node in np.ndindex(8, 15):
        block_avg_patterns_loc[block, node, :] = zstats_loc_lss[(nodes_lss == node) & (blocks_lss == block), :nvoxels].mean(0)
    block_avg_results_loc = np.zeros((15, 15))
    for node_i, node_j, block_i, block_j in np.ndindex(15, 15, 8, 8):
        if block_i != block_j:
            block_avg_results_loc[node_i, node_j] += np.corrcoef(block_avg_patterns_loc[block_i, node_i, :],
                                                                 block_avg_patterns_loc[block_j, node_j, :])[0][1]
    block_avg_results_loc /= 8 * 7
    np.save(f'{output_dir}/sub-{subject_id}_rescale-{rescale}_subtract-mean-{subtract_mean}_cross-val-corr_{roi_name}_zstats_{nvoxels}_block-avg.npy', block_avg_results_loc, allow_pickle=False)

    del zstats_loc_lss
    gc.collect()

for (rescale, subtract_mean) in zstat_params:
    rois = (
        ('hippocampus-both', partial(load_hippocampus, hemi='both')),
        ('hippocampus-lh', partial(load_hippocampus, hemi='left')),
        ('hippocampus-rh', partial(load_hippocampus, hemi='right')),
        ('entorhinal-both', partial(load_entorhinal, hemi='both')),
        ('entorhinal-lh', partial(load_entorhinal, hemi='left')),
        ('entorhinal-rh', partial(load_entorhinal, hemi='right')),
        ('lingual-both', partial(load_lingual, hemi='both')),
        ('lingual-lh', partial(load_lingual, hemi='left')),
        ('lingual-rh', partial(load_lingual, hemi='right')),
        ('pericalcarine-both', partial(load_pericalcarine, hemi='both')),
        ('pericalcarine-lh', partial(load_pericalcarine, hemi='left')),
        ('pericalcarine-rh', partial(load_pericalcarine, hemi='right')),
        ('cuneus-both', partial(load_cuneus, hemi='both')),
        ('cuneus-lh', partial(load_cuneus, hemi='left')),
        ('cuneus-rh', partial(load_cuneus, hemi='right')),
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

        logging.info(f'Loading z-stats {roi_name} LS-S...')
        zstats_lss = load_zstats_lss(mask_roi, rescale=rescale, subtract_mean=subtract_mean)
        logging.info(f'{zstats_lss.shape[0]} events')
        logging.info(f'{zstats_lss.shape[1]} voxels')

        block_avg_patterns = np.zeros((8, 15, zstats_lss.shape[1]))
        for block, node in np.ndindex(8, 15):
            block_avg_patterns[block, node, :] = zstats_lss[(nodes_lss == node) & (blocks_lss == block), :].mean(0)
        block_avg_results = np.zeros((15, 15))
        for node_i, node_j, block_i, block_j in np.ndindex(15, 15, 8, 8):
            if block_i != block_j:
                block_avg_results[node_i, node_j] += np.corrcoef(block_avg_patterns[block_i, node_i, :],
                                                                 block_avg_patterns[block_j, node_j, :])[0][1]
        block_avg_results /= 8 * 7
        np.save(f'{output_dir}/sub-{subject_id}_rescale-{rescale}_subtract-mean-{subtract_mean}_cross-val-corr_{roi_name}_zstats_block-avg.npy', block_avg_results, allow_pickle=False)
        gc.collect()

    # corr_mat_loc = np.zeros((15, 15))
    # corr_mat_loc_n = np.zeros((15, 15))
    # nvoxels = 800
    # for node_idx_i in range(len(nodes_lss)):
    #     for node_type_j in range(15):
    #         node_type_i = nodes_lss[node_idx_i]
    #         block_i = blocks_lss[node_idx_i]
    #         node_i_zstats = zstats_loc_lss[node_idx_i, :nvoxels]
    #         node_j_zstats = zstats_loc_lss[(nodes_lss == node_type_j) & (blocks_lss != block_i), :nvoxels]
    #         corrs = np.corrcoef(node_i_zstats, node_j_zstats)[0, 1:]
    #         corr_mat_loc[node_type_i, node_type_j] += corrs.sum()
    #         corr_mat_loc_n[node_type_i, node_type_j] += len(corrs)
    # cross_val_corr_loc = corr_mat_loc / corr_mat_loc_n
    # np.save(f'{output_dir}/sub-{subject_id}_cross-val-corr_loc_zstats_{nvoxels}_event-avg.npy', cross_val_corr_loc, allow_pickle=False)
    #
    # corr_mat_hipp = np.zeros((15, 15))
    # corr_mat_hipp_n = np.zeros((15, 15))
    # for node_idx_i in range(len(nodes_lss)):
    #     for node_type_j in range(15):
    #         node_type_i = nodes_lss[node_idx_i]
    #         block_i = blocks_lss[node_idx_i]
    #         node_i_zstats = zstats_hipp_lss[node_idx_i, :]
    #         node_j_zstats = zstats_hipp_lss[(nodes_lss == node_type_j) & (blocks_lss != block_i), :]
    #         corrs = np.corrcoef(node_i_zstats, node_j_zstats)[0, 1:]
    #         corr_mat_hipp[node_type_i, node_type_j] += corrs.sum()
    #         corr_mat_hipp_n[node_type_i, node_type_j] += len(corrs)
    # cross_val_corr_hipp = corr_mat_hipp / corr_mat_hipp_n
    # np.save(f'{output_dir}/sub-{subject_id}_cross-val-corr_hipp_zstats_event-avg.npy', cross_val_corr_hipp, allow_pickle=False)
    #
    # corr_mat_ento = np.zeros((15, 15))
    # corr_mat_ento_n = np.zeros((15, 15))
    # for node_idx_i in range(len(nodes_lss)):
    #     for node_type_j in range(15):
    #         node_type_i = nodes_lss[node_idx_i]
    #         block_i = blocks_lss[node_idx_i]
    #         node_i_zstats = zstats_ento_lss[node_idx_i, :]
    #         node_j_zstats = zstats_ento_lss[(nodes_lss == node_type_j) & (blocks_lss != block_i), :]
    #         corrs = np.corrcoef(node_i_zstats, node_j_zstats)[0, 1:]
    #         corr_mat_ento[node_type_i, node_type_j] += corrs.sum()
    #         corr_mat_ento_n[node_type_i, node_type_j] += len(corrs)
    # cross_val_corr_ento = corr_mat_ento / corr_mat_ento_n
    # np.save(f'{output_dir}/sub-{subject_id}_cross-val-corr_ento_zstats_event-avg.npy', cross_val_corr_ento, allow_pickle=False)
