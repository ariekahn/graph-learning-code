from nilearn.image import load_img, math_img, index_img, new_img_like, mean_img
from nilearn import plotting

# from mvpa2.mappers.zscore import zscore
from mvpa2.measures import rsa
from mvpa2.base.dataset import vstack
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.measures.searchlight import sphere_searchlight
from mvpa2.suite import map2nifti
from mvpa2.base.learner import ChainLearner
from mvpa2.mappers.shape import TransposeMapper

import argparse
import os
import os.path as op
import errno
import logging

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
import numpy as np


# exist_ok
def makedirs(folder, *args, **kwargs):
    try:
        return os.makedirs(folder, exist_ok=True, *args, **kwargs)
    except TypeError:  # Unexpected arguments encountered
        pass

    try:
        # Should work is TypeError was caused by exist_ok, eg., Py2
        return os.makedirs(folder, *args, **kwargs)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

        if op.isfile(folder):
            # folder is a file, raise OSError just like os.makedirs() in Py3
            raise


parser = argparse.ArgumentParser('Run MVPA analysis')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('project_dir', type=str, help='directory containing data and derived subfolders')
parser.add_argument('output_dir', type=str, help='Location for output files')
parser.add_argument('radius', type=int, help='Searchlight radius')
args = parser.parse_args()

subject = args.subject_id
project_dir = args.project_dir
output_dir = args.output_dir
radius = args.radius

# output_dir = op.join(project_dir, 'derived/pymvpa-' + str(datetime.date.today()))
image_dir = op.join(output_dir, 'images')
makedirs(output_dir)
makedirs(image_dir)
fmriprep_dir = op.join(project_dir, 'derived/fmriprep')
lss_dir = op.join(project_dir, 'derived/feat_representation_lss')
data_dir = op.join(project_dir, 'data')

logging.basicConfig(filename='{}/pymvpa_searchlight_{}.log'.format(output_dir, subject), filemode='w', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info('Subject: {}'.format(subject))


_modular_list = {
    0: [1, 2, 3, 14],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [1, 2, 3, 5],
    5: [4, 6, 7, 8],
    6: [5, 7, 8, 9],
    7: [5, 6, 8, 9],
    8: [5, 6, 7, 9],
    9: [6, 7, 8, 10],
    10: [9, 11, 12, 13],
    11: [10, 12, 13, 14],
    12: [10, 11, 13, 14],
    13: [10, 11, 12, 14],
    14: [11, 12, 13, 0],
}

_ring_lattice_list = {
    0: [13, 14, 1, 2],
    1: [14, 0, 2, 3],
    2: [0, 1, 3, 4],
    3: [1, 2, 4, 5],
    4: [2, 3, 5, 6],
    5: [3, 4, 6, 7],
    6: [4, 5, 7, 8],
    7: [5, 6, 8, 9],
    8: [6, 7, 9, 10],
    9: [7, 8, 10, 11],
    10: [8, 9, 11, 12],
    11: [9, 10, 12, 13],
    12: [10, 11, 13, 14],
    13: [11, 12, 14, 0],
    14: [12, 13, 0, 1],
}

modular = nx.from_dict_of_lists(_modular_list)
ring_lattice = nx.from_dict_of_lists(_ring_lattice_list)
comm_modular = pd.DataFrame(nx.communicability(modular)).values
comm_lattice = pd.DataFrame(nx.communicability(ring_lattice)).values

dissim_lattice = np.max(comm_lattice) - comm_lattice
dissim_lattice = dissim_lattice.round(3)
np.fill_diagonal(dissim_lattice, 0)

dissim_modular = np.max(comm_modular) - comm_modular
dissim_modular = dissim_modular.round(3)
np.fill_diagonal(dissim_modular, 0)


def load_aparc(subject, run):
    """
    Load Freesurfer aparc for graph representation run

    derived/fmriprep/sub-{subject}/ses-2/func/sub-{subject}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz
    """
    aparc = load_img('{}/sub-{}/ses-2/func/sub-{}_ses-2_task-graphrepresentation_run-{}_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz'.format(
        fmriprep_dir, subject, subject, run))
    return aparc


def load_boldref(subject, run):
    """
    Load Freesurfer aparc for graph representation run

    derived/fmriprep/sub-{subject}/ses-2/func/sub-{subject}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz
    """
    boldref = load_img('{}/sub-{}/ses-2/func/sub-{}_ses-2_task-graphrepresentation_run-{}_space-MNI152NLin2009cAsym_boldref.nii.gz'.format(
        fmriprep_dir, subject, subject, run))
    return boldref


def load_loc(subject, run, hemi='both'):
    """
    Load a mask for LOC

    subject : str
        GLSxxx
    run : int
        1-8
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

    aparc = load_aparc(subject, run)
    loc = math_img(roi_str, img=aparc)
    return loc


def load_hippocampus(subject, run):
    roi_str = '(img == 17) | (img == 53)'

    aparc = load_aparc(subject, run)
    hipp = math_img(roi_str, img=aparc)
    return hipp


def load_entorhinal(subject, run):
    roi_str = '(img == 1006) | (img == 2006)'

    aparc = load_aparc(subject, run)
    ento = math_img(roi_str, img=aparc)
    return ento


def load_full_mask(subject, run):
    roi_str = '(img == 1011) | (img == 2011) | (img == 17) | (img == 53) | (img == 1006) | (img == 2006)'

    aparc = load_aparc(subject, run)
    mask = math_img(roi_str, img=aparc)
    return mask


def load_bold_mask(subject):
    """
    Load bold mask intersection across all runs

    derived/fmriprep/sub-{subject}/ses-2/func/sub-{subject}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
    """
    boldmask = load_img('{}/sub-{}/ses-2/func/sub-{}_ses-2_task-graphrepresentation_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(
        fmriprep_dir, subject, subject))
    return boldmask


def load_masked_template(subject):
    """
    Load the T1w template, and mask it

    derived/fmriprep/sub-{subject}/anat/sub-{subject}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz

    Usage:
    template = load_masked_template('GLS011')
    """
    # Load T1w template and mask
    template = load_img('{}/sub-{}/anat/sub-{}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'.format(fmriprep_dir, subject, subject))
    template_mask = load_img('{}/sub-{}/anat/sub-{}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(fmriprep_dir, subject, subject))
    template = math_img("img1 * img2", img1=template, img2=template_mask)
    return template


# def load_ds_lsa(subject):
#     ds_sets = []
#     mask = load_bold_mask(subject)
#     for run in range(1, 9):
#
#         # Load the zstat for each event
#         nodes = range(1, 16)
#         zstats = []
#         for node in nodes:
#             zstat_file = op.join(lsa_dir, 'sub-{}/run-{}/zfiles/fwhm-5.0_zstat{}.nii.gz'.format(subject, run, node))
#             zstats.append(zstat_file)
#         ds_sets.append(fmri_dataset(zstats, targets=nodes, chunks=run, mask=mask))
#
#     # Stack all runs
#     ds = vstack(ds_sets, a=0)
#
#     # Remove any non-varying voxels
#     ds = ds[:, ds.samples.std(0) > 0]
#
#     return ds


def load_ds_lss(subject):
    ds_sets = []
    mask = load_bold_mask(subject)
    for run in range(1, 9):
        zstats = []

        events_file = op.join(data_dir, 'sub-{}/ses-2/func/sub-{}_ses-2_task-graphrepresentation_run-{}_events.tsv'.format(subject, subject, run))
        events = pd.read_csv(events_file, sep='\t')
        events_loaded = []
        for event in range(1, 60):
            zstat_file = op.join(lss_dir, 'sub-{}/run-{}/zfiles/fwhm-5.0_event-{}_zstat1.nii.gz'.format(subject, run, event))
            zstats.append(zstat_file)
            events_loaded.append(event)
            # if op.exists(zstat_file):
            #     zstats.append(zstat_file)
            #     events_loaded.append(event)
            # else:
            #     print('Missing: ' + zstat_file)
        nodes = events.node[events_loaded].values
        ds_sets.append(fmri_dataset(zstats, targets=nodes, chunks=run, mask=mask))

    # Stack all runs
    ds = vstack(ds_sets, a=0)

    # Remove any non-varying voxels
    ds = ds[:, ds.samples.std(0) > 0]

    return ds


# little helper function to plot dissimilarity matrices
# since we are using correlation-distance, we use colorbar range of [0,2]
def plot_mtx(mtx, labels, title):
    plt.figure()
    plt.imshow(mtx, interpolation='nearest')
    plt.xticks(range(len(mtx)), range(15), rotation=-45)
    plt.yticks(range(len(mtx)), range(150))
    plt.title(title)
    plt.clim((0, 2))
    plt.colorbar()


def run_searchlight(ds, template, radius, pairwise_metric):
    """Compute the RDM across a searchlight
    """
    # Mean Target Group Sample
    mtgs = mean_group_sample(['targets'])
    # Mean Target Dataset
    mtds = mtgs(ds)
    dsm = rsa.PDist(square=False, pairwise_metric=pairwise_metric)
    sl = sphere_searchlight(dsm, radius)
    slres = sl(mtds)

    # Examine results
    niftiresults = map2nifti(slres, imghdr=ds.a.imghdr)
    # Fix affine
    niftiresults._affine = ds.a.imgaffine
    niftiresults.to_filename('{}/sub-{}_searchlight-lss_pairwise-metric-{}_radius-{}_desc-rdm.nii.gz'.format(sub_result_dir, subject, pairwise_metric, radius))

    """ Pattern dissimilarity
    """
    # score each searchlight sphere result wrt global pattern dissimilarity
    distinctiveness = np.sum(np.abs(slres), axis=0)
    most_distinct_loc = mtds.fa.voxel_indices[distinctiveness.argmax()]
    # logging.info('Most dissimilar patterns around' +
    #       str(most_distinct_loc))
    # take a look at the this dissimilarity structure
    distinct_mat = squareform(slres.samples[:, distinctiveness.argmax()])
    filename = '{}/sub-{}_searchlight-lss_pairwise-metric-{}_radius-{}_desc-most-distinct-rdm.npy'.format(sub_mat_dir, subject, pairwise_metric, radius)
    np.save(file=filename, arr=distinct_mat, allow_pickle=False)
    plot_mtx(distinct_mat,
             mtds.sa.targets,
             'Maximum distinctive searchlight pattern correlation distances\n{}, {}'.format(subject, most_distinct_loc))
    figname = 'sub-{}_searchlight-lss_pairwise-metric-{}_radius-{}_desc-most-distinct-rdm.pdf'.format(subject, pairwise_metric, radius)
    plt.tight_layout()
    plt.savefig(op.join(sub_image_dir, figname))

    distinct_map = new_img_like(niftiresults, np.mean(np.abs(niftiresults.get_fdata()), 3), niftiresults.affine)
    distinct_map.to_filename('{}/sub-{}_searchlight-lss_pairwise-metric-{}_radius-{}_desc-distinct-map.nii.gz'.format(sub_result_dir, subject, pairwise_metric, radius))

    # Change the value range so it can be more easily plotted
    distinct_map_corrected = math_img('((img - min(img[img>0])) * 10) * (img > 0)', img=distinct_map)
    plotting.plot_stat_map(distinct_map_corrected,
                           bg_img=template, black_bg=False, title='Distinctiveness Map (Altered Range) {}'.format(subject))
    figname = op.join(sub_image_dir, 'sub-{}_searchlight-lss_pairwise-metric-{}_radius-{}_desc-distinct-map.pdf'.format(subject, pairwise_metric, radius))
    plt.savefig(figname)

    for comparison_metric in ['pearson', 'spearman']:
        logging.info(comparison_metric)
        """ Consistency
        Let's look at the stability of similarity structures
        across experiment runs
        mean condition samples, as before, but now individually for each run
        """
        mtcgs = mean_group_sample(['targets', 'chunks'])
        mtcds = mtcgs(ds)

        # searchlight consistency measure -- how correlated are the structures
        # across runs
        dscm = rsa.PDistConsistency(pairwise_metric=pairwise_metric, consistency_metric=comparison_metric)
        sl_cons = sphere_searchlight(dscm, radius)
        slres_cons = sl_cons(mtcds)

        # mean correlation
        mean_consistency = np.mean(slres_cons, axis=0)
        # Replace NAs
        np.nan_to_num(mean_consistency)
        most_consistent_loc = mtds.fa.voxel_indices[mean_consistency.argmax()]
        # logging.info('Most stable dissimilarity patterns around'.format())
        # Look at this pattern
        consistent_mat = squareform(slres.samples[:, mean_consistency.argmax()])
        filename = '{}/sub-{}_searchlight-lss_pairwise-metric-{}_consistency-metric-{}_radius-{}_desc-most-consistent-rdm.npy'.format(sub_mat_dir, subject, pairwise_metric, comparison_metric, radius)
        np.save(file=filename, arr=consistent_mat, allow_pickle=False)
        plot_mtx(consistent_mat,
                 mtds.sa.targets,
                 'Most consistent searchlight pattern correlation distances\n{}, {}'.format(subject, most_consistent_loc))
        figname = 'sub-{}_searchlight-lss_pairwise-metric-{}_consistency-metric-{}_radius-{}_desc-most-consistent-rdm.pdf'.format(subject, pairwise_metric, comparison_metric, radius)
        plt.tight_layout()
        plt.savefig(op.join(sub_image_dir, figname))

        consistency_map = map2nifti(slres_cons, imghdr=ds.a.imghdr)
        # Fix affine
        consistency_map._affine = ds.a.imgaffine
        consistency_map.to_filename('{}/sub-{}_searchlight-lss_pairwise-metric-{}_consistency-metric-{}_radius-{}_desc-consistency-map.nii.gz'.format(sub_result_dir, subject, pairwise_metric, comparison_metric, radius))

        plotting.plot_stat_map(mean_img(consistency_map), bg_img=template, black_bg=False, title='Consistency Map {}'.format(subject))
        figname = op.join(sub_image_dir, 'sub-{}_searchlight-lss_pairwise-metric-{}_consistency-metric-{}_radius-{}_desc-consistency-map.pdf'.format(subject, pairwise_metric, comparison_metric, radius))
        plt.savefig(figname)

        """Lattice/Modular Comparison
        """
        # let's see where in the brain we find dissimilarity structures that are
        # similar to our most stable one
        tdsm_modular = rsa.PDistTargetSimilarity(squareform(dissim_modular), pairwise_metric=pairwise_metric, comparison_metric=comparison_metric)
        # using a searchlight

        sl_tdsm_modular = sphere_searchlight(ChainLearner([tdsm_modular, TransposeMapper()]), radius)
        slres_tdsm_modular = sl_tdsm_modular(mtds)
        modular_map = map2nifti(slres_tdsm_modular, imghdr=ds.a.imghdr)
        modular_map._affine = ds.a.imgaffine

        modular_map_rho = index_img(modular_map, 0)
        modular_map_pval = index_img(modular_map, 1)
        modular_map_rho.to_filename('{}/sub-{}_searchlight-lss_pairwise-metric-{}_comparison-metric-{}_radius-{}_desc-modular-map-rho.nii.gz'.format(sub_result_dir, subject, pairwise_metric, comparison_metric, radius))
        modular_map_pval.to_filename('{}/sub-{}_searchlight-lss_pairwise-metric-{}_comparison-metric-{}_radius-{}_desc-modular-map-pval.nii.gz'.format(sub_result_dir, subject, pairwise_metric, comparison_metric, radius))

        plotting.plot_stat_map(modular_map_rho, bg_img=template, black_bg=False, title='Modular rho {}'.format(subject))
        figname = op.join(sub_image_dir, 'sub-{}_searchlight-lss_pairwise-metric-{}_comparison-metric-{}_radius-{}_desc-modular-rho.pdf'.format(subject, pairwise_metric, comparison_metric, radius))
        plt.savefig(figname)
        plotting.plot_stat_map(modular_map_pval, bg_img=template, black_bg=False, title='Modular p-val {}'.format(subject))
        figname = op.join(sub_image_dir, 'sub-{}_searchlight-lss_pairwise-metric-{}_comparison-metric-{}_radius-{}_desc-modular-pval.pdf'.format(subject, pairwise_metric, comparison_metric, radius))
        plt.savefig(figname)

        # Lattice
        tdsm_lattice = rsa.PDistTargetSimilarity(squareform(dissim_lattice), pairwise_metric=pairwise_metric, comparison_metric=comparison_metric)
        sl_tdsm_lattice = sphere_searchlight(ChainLearner([tdsm_lattice, TransposeMapper()]), radius)
        slres_tdsm_lattice = sl_tdsm_lattice(mtds)
        lattice_map = map2nifti(slres_tdsm_lattice, imghdr=ds.a.imghdr)
        lattice_map._affine = ds.a.imgaffine

        lattice_map_rho = index_img(lattice_map, 0)
        lattice_map_pval = index_img(lattice_map, 1)
        lattice_map_rho.to_filename('{}/sub-{}_searchlight-lss_pairwise-metric-{}_comparison-metric-{}_radius-{}_desc-lattice-map-rho.nii.gz'.format(sub_result_dir, subject, pairwise_metric, comparison_metric, radius))
        lattice_map_pval.to_filename('{}/sub-{}_searchlight-lss_pairwise-metric-{}_comparison-metric-{}_radius-{}_desc-lattice-map-pval.nii.gz'.format(sub_result_dir, subject, pairwise_metric, comparison_metric, radius))

        plotting.plot_stat_map(lattice_map_rho, bg_img=template, black_bg=False, title='Lattice rho {}'.format(subject))
        figname = op.join(sub_image_dir, 'sub-{}_searchlight-lss_pairwise-metric-{}_comparison-metric-{}_radius-{}_desc-lattice-rho.pdf'.format(subject, pairwise_metric, comparison_metric, radius))
        plt.savefig(figname)
        plotting.plot_stat_map(lattice_map_pval, bg_img=template, black_bg=False, title='Lattice p-val {}'.format(subject))
        figname = op.join(sub_image_dir, 'sub-{}_searchlight-lss_pairwise-metric-{}_comparison-metric-{}_radius-{}_desc-lattice-pval.pdf'.format(subject, pairwise_metric, comparison_metric, radius))
        plt.savefig(figname)


# Load the subject
logging.info(subject)
sub_image_dir = op.join(output_dir, 'images')
sub_mat_dir = op.join(output_dir, 'mats')
sub_result_dir = op.join(output_dir, 'results')
makedirs(sub_image_dir)
makedirs(sub_result_dir)
makedirs(sub_mat_dir)

ds = load_ds_lss(subject)
# zscore(ds, chunks_attr='chunks')
template = load_masked_template(subject)
for pairwise_metric in ['euclidean', 'correlation']:
    logging.info(pairwise_metric)
    run_searchlight(ds, template, radius, pairwise_metric)
