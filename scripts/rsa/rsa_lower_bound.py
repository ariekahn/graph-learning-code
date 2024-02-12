import argparse
import logging
import gc
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from nilearn.image import load_img, new_img_like

parser = argparse.ArgumentParser('Run MVPA analysis')
parser.add_argument('project_dir', type=str, help='directory containing data and derived subfolders')
parser.add_argument('output_dir', type=str, help='Location for output files')
parser.add_argument('ordering', type=str, choices=('node', 'movement', 'shape'), help='Ordering for cross-subject comparison')
parser.add_argument('--zscored', action='store_true', help='Use zscored RDM computations')
args = parser.parse_args()

project_dir = args.project_dir
output_dir = args.output_dir
ordering = args.ordering
zscored = args.zscored

logging.basicConfig(filename=f'{output_dir}/rsa_lower-bound_ordering-{ordering}_zscored-{zscored}.log', filemode='w', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

data_dir = f'{project_dir}/data'

subjects = ['GLS003', 'GLS004', 'GLS005', 'GLS006',
            'GLS008', 'GLS009', 'GLS010', 'GLS011',
            'GLS013', 'GLS014', 'GLS017', 'GLS018',
            'GLS019', 'GLS020', 'GLS021', 'GLS022',
            'GLS023', 'GLS024', 'GLS025', 'GLS026',
            'GLS027', 'GLS028', 'GLS030', 'GLS033',
            'GLS037', 'GLS038', 'GLS039', 'GLS040',
            'GLS043', 'GLS044', 'GLS045']

graph = (('GLS003', 'Modular'), ('GLS004', 'Modular'), ('GLS005', 'Lattice'), ('GLS006', 'Lattice'),
         ('GLS008', 'Modular'), ('GLS009', 'Lattice'), ('GLS010', 'Modular'), ('GLS011', 'Modular'),
         ('GLS013', 'Modular'), ('GLS014', 'Lattice'), ('GLS016', 'Modular'), ('GLS017', 'Lattice'),
         ('GLS018', 'Modular'), ('GLS019', 'Modular'), ('GLS020', 'Lattice'), ('GLS021', 'Lattice'),
         ('GLS022', 'Modular'), ('GLS023', 'Modular'), ('GLS024', 'Lattice'), ('GLS025', 'Modular'),
         ('GLS026', 'Modular'), ('GLS027', 'Lattice'), ('GLS028', 'Lattice'), ('GLS030', 'Lattice'),
         ('GLS033', 'Lattice'), ('GLS034', 'Lattice'), ('GLS036', 'Modular'), ('GLS037', 'Modular'),
         ('GLS038', 'Modular'), ('GLS039', 'Modular'), ('GLS040', 'Lattice'), ('GLS041', 'Lattice'),
         ('GLS043', 'Lattice'), ('GLS044', 'Modular'), ('GLS045', 'Lattice'),
         )
graph_df = pd.DataFrame(graph, columns=('subject', 'graph'))
graph_df = graph_df.loc[graph_df.subject.isin(subjects)].reset_index(drop=True)

subjects_modular = list(graph_df[graph_df['graph'] == 'Modular'].subject.values)
subjects_modular_idx = graph_df[graph_df['graph'] == 'Modular'].subject.index.values
subjects_lattice = list(graph_df[graph_df['graph'] == 'Lattice'].subject.values)
subjects_lattice_idx = graph_df[graph_df['graph'] == 'Lattice'].subject.index.values


results = {}
for subject in subjects:
    if zscored:
        img = load_img(f'{project_dir}/derived/cosmo_make_rdm_searchlight/sub-{subject}/results/sub-{subject}_rdm_nvox-100_searchlight-lss-zfiles-zscored.nii.gz')
    else:
        img = load_img(f'{project_dir}/derived/cosmo_make_rdm_searchlight/sub-{subject}/results/sub-{subject}_rdm_nvox-100_searchlight-lss-zfiles.nii.gz')
    results[subject] = img
# For later recreating the images
ref_img = results[subjects[0]]

if ordering == 'shape':
    node_to_shape = dict()  # [shape0, shape1, etc]
    shape_to_node = dict()
    node_to_shape_square = dict()  # [shape0, shape1, etc]
    shape_to_node_square = dict()
    for subject in subjects:
        filename = f'{data_dir}/sub-{subject}/ses-2/func/sub-{subject}_ses-2_task-graphrepresentation_run-1_events.tsv'
        data = pd.read_csv(filename, sep='\t')
        node_to_shape[subject] = data.groupby('shape', sort=True)['node'].first().values
        shape_to_node[subject] = np.argsort(node_to_shape[subject])
        node_to_shape_square[subject] = squareform(squareform(range(105))[node_to_shape[subject], :][:, node_to_shape[subject]])
        shape_to_node_square[subject] = squareform(squareform(range(105))[shape_to_node[subject], :][:, shape_to_node[subject]])
        results[subject] = new_img_like(results[subject],
                                        results[subject].get_fdata()[:, :, :, node_to_shape_square[subject]],
                                        affine=results[subject].affine,
                                        copy_header=True)
elif ordering == 'movement':
    movement_order = [
        '[10000]',
        '[01000]',
        '[00100]',
        '[00010]',
        '[00001]',
        '[11000]',
        '[10100]',
        '[10010]',
        '[10001]',
        '[01100]',
        '[01010]',
        '[01001]',
        '[00110]',
        '[00101]',
        '[00011]',
    ]
    node_to_movement = dict()
    movement_to_node = dict()
    node_to_movement_square = dict()
    movement_to_node_square = dict()
    for subject in subjects:
        filename = f'{data_dir}/sub-{subject}/ses-2/func/sub-{subject}_ses-2_task-graphrepresentation_run-1_events.tsv'
        data = pd.read_csv(filename, sep='\t')
        node_to_movement[subject] = data.groupby('movement_correct', sort=True)['node'].first()[movement_order].values
        movement_to_node[subject] = np.argsort(node_to_movement[subject])
        node_to_movement_square[subject] = squareform(squareform(range(105))[node_to_movement[subject], :][:, node_to_movement[subject]])
        movement_to_node_square[subject] = squareform(squareform(range(105))[movement_to_node[subject], :][:, movement_to_node[subject]])
        results[subject] = new_img_like(results[subject],
                                        results[subject].get_fdata()[:, :, :, node_to_movement_square[subject]],
                                        affine=results[subject].affine,
                                        copy_header=True)
elif ordering == 'node':
    pass
else:
    raise ValueError('Invalid matrix ordering')

n_voxels = 81 * 96 * 81
data_flat_all = np.stack([results[subject].get_fdata().reshape(n_voxels, 105) for subject in subjects])
del results
gc.collect()

data_flat_modular = data_flat_all[subjects_modular_idx, :, :]
data_flat_lattice = data_flat_all[subjects_lattice_idx, :, :]

logging.info('Modular')
results_modular = np.zeros(n_voxels)
for subject_ind in range(len(subjects_modular)):
    logging.info(f'{subject_ind}')
    x1 = np.concatenate([1 - data_flat_modular[0:subject_ind, :, :], 1 - data_flat_modular[subject_ind + 1:, :, :]]).mean(0)
    x2 = 1 - data_flat_modular[subject_ind, :, :]
    for voxel in range(n_voxels):
        results_modular[voxel] += np.corrcoef(x1[voxel, :], x2[voxel, :])[0][1]
results_modular /= len(subjects_modular)
results_modular_img = new_img_like(ref_img, np.nan_to_num(results_modular.reshape(81, 96, 81)))
results_modular_img.to_filename(f'{output_dir}/rdm_lower-bound_zscored-{zscored}_ordering-{ordering}_modular.nii.gz')
gc.collect()

logging.info('Lattice')
results_lattice = np.zeros(n_voxels)
for subject_ind in range(len(subjects_lattice)):
    logging.info(f'{subject_ind}')
    x1 = np.concatenate([1 - data_flat_lattice[0:subject_ind, :, :], 1 - data_flat_lattice[subject_ind + 1:, :, :]]).mean(0)
    x2 = 1 - data_flat_lattice[subject_ind, :, :]
    for voxel in range(n_voxels):
        results_lattice[voxel] += np.corrcoef(x1[voxel, :], x2[voxel, :])[0][1]
results_lattice /= len(subjects_lattice)
results_lattice_img = new_img_like(ref_img, np.nan_to_num(results_lattice.reshape(81, 96, 81)))
results_lattice_img.to_filename(f'{output_dir}/rdm_lower-bound_zscored-{zscored}_ordering-{ordering}_lattice.nii.gz')
gc.collect()

logging.info('All')
results_all = np.zeros(n_voxels)
for subject_ind in range(len(subjects)):
    logging.info(f'{subject_ind}')
    x1 = np.concatenate([1 - data_flat_all[0:subject_ind, :, :], 1 - data_flat_all[subject_ind + 1:, :, :]]).mean(0)
    x2 = 1 - data_flat_all[subject_ind, :, :]
    for voxel in range(n_voxels):
        results_all[voxel] += np.corrcoef(x1[voxel, :], x2[voxel, :])[0][1]
results_all /= len(subjects)
results_all_img = new_img_like(ref_img, np.nan_to_num(results_all.reshape(81, 96, 81)))
results_all_img.to_filename(f'{output_dir}/rdm_lower-bound_zscored-{zscored}_ordering-{ordering}_all.nii.gz')
gc.collect()
