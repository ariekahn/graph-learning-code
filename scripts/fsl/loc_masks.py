import argparse
import logging
import os.path as op
import numpy as np
import pandas as pd
from scipy import ndimage
from nilearn.image import load_img, math_img, new_img_like

parser = argparse.ArgumentParser('Create LOC masks')
parser.add_argument('fmriprep_dir', type=str, help='fmriprep directory base')
parser.add_argument('feat_dir', type=str, help='Location of FEAT analysis')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('n_voxels', type=int, help='Number of voxels to include in the mask')
parser.add_argument('--dilate', type=int, default=0, help='Number of voxels to dilate mask')
parser.add_argument('-d', '--dry_run', action='store_true', help="Don't write changes.")
parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output.')
args = parser.parse_args()

subject_id = args.subject_id
fmriprep_dir = op.abspath(args.fmriprep_dir)
feat_dir = op.abspath(args.feat_dir)
dilation = args.dilate
n_voxels = args.n_voxels

if args.verbose:
    logging.basicConfig(level=logging.INFO)

logging.info(f'Subject: {subject_id}')
logging.info(f'fMRIPrep Dir: {fmriprep_dir}')
logging.info(f'FEAT Dir: {feat_dir}')
logging.info(f'n_voxels: {n_voxels}')
logging.info(f'Dilation: {dilation}')

def make_loc_masks(localizer, aparc, n_voxels, dilation=0):
    """
    localizer: FEAT zstat nifti
    aparc: fmriprep aparcaseg
    n_voxels: how many top voxels to preserve
    dilation: how many voxels to dilate by
    """
    # We want 2011 ctx-rh-lateraloccipital
    # and 1011 ctx-lh-lateraloccipital

    roi_vals = dict(lh=1011, rh=2011)
    masks = {}

    for hemi in ['lh', 'rh']:
        # Load the fmriprep aparc and pull out the appropriate regions
        loc_aparc = math_img(f'img == {roi_vals[hemi]}', img=aparc)

        if dilation > 0:
            dil_values = ndimage.binary_dilation(loc_aparc.get_fdata(), iterations=dilation)
            loc_aparc = new_img_like(
                loc_aparc,
                dil_values.astype(np.int))
            # But mask it to stay inside the brain
            loc_aparc = math_img('img1 * (img2 > 0)', img1=loc_aparc, img2=aparc)

        # Mask the localizer values with the aparc region
        loc_raw = math_img("img1 * img2", img1=loc_aparc, img2=localizer)

        # Find the z-score cutoff 
        cutoff = np.sort(loc_raw.get_fdata().flatten())[-n_voxels]

        # Compute the a mask of the top n voxels
        loc_mask = math_img(f'img >= {cutoff}', img=loc_raw)

        masks[hemi] = loc_mask
    return masks

localizer = load_img(f'{feat_dir}/sub-{subject_id}/localizer.feat/stats/zstat1.nii.gz')
aparc = load_img(f'{fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin6Asym_res-2_desc-aparcaseg_dseg.nii.gz')
masks = make_loc_masks(localizer, aparc, n_voxels, dilation=dilation)

for hemi in ['lh', 'rh']:
    masked = math_img("img1 * img2", img1=localizer, img2=masks[hemi])

    out_name = f'{feat_dir}/sub-{subject_id}/localizer.feat/stats/zstat1_masked_{hemi}.nii.gz'
    logging.info(out_name)
    if not args.dry_run:
        masked.to_filename(out_name)

for n in range(1,16):
    data = load_img(f'{feat_dir}/sub-{subject_id}/level-2.gfeat/cope{n}.feat/stats/zstat1.nii.gz')
    for hemi in ['lh', 'rh']:
        masked = math_img("img1 * img2", img1=data, img2=masks[hemi])
        n_voxels_found = (np.abs(masked.get_fdata()) > 0).sum()
        if n_voxels_found < n_voxels:
            logging.warning(f'{n_voxels_found} voxels found for subject {subject_id} cope {n}')

        out_name = f'{feat_dir}/sub-{subject_id}/level-2.gfeat/cope{n}.feat/stats/zstat1_masked_{hemi}.nii.gz'
        logging.info(out_name)
        if not args.dry_run:
            masked.to_filename(out_name)
