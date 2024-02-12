function cosmo_make_rdm_searchlight(subject, project_path, output_path, nproc)
%% Check variables exist
% 

if ~exist('subject', 'var')
    disp('subject missing');
    exit();
end

if ~exist('project_path', 'var')
    disp('project_path missing');
    exit();
end

if ~exist('output_path', 'var')
    disp('output_path missing');
    exit();
end

nproc = str2num(nproc);

diary(fullfile(output_path, sprintf('sub-%s.txt', subject)));

disp(subject);

nruns = 8;
% nevents = 59;

%% Paths
fmriprep_func_path = fullfile(project_path, sprintf('derived/fmriprep/sub-%s/ses-2/func', subject));
event_path = fullfile(project_path, sprintf('data/sub-%s/ses-2/func', subject));
lss_path = fullfile(project_path, sprintf('derived/feat_representation_lss/sub-%s', subject));
%     lsa_path = fullfile(project_path, sprintf('derived/feat_representation_lsa/sub-%s', subject));
%% Combined mask
bold_mask_path = fullfile(fmriprep_func_path, sprintf('sub-%s_ses-2_task-graphrepresentation_run-%d_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz', subject, 1));
load_nii(bold_mask_path); % Only here to get mcc to include fn - need to fix
%% Load Data
disp('Loading LS-S data');
disp('zfiles');
ds_lss_zfiles_all = cell(59 * nruns, 1);
counter = 0;
for runidx = 1:nruns
    events_file = fullfile(event_path, sprintf('sub-%s_ses-2_task-graphrepresentation_run-%d_events.tsv', subject, runidx));

    % avoid statistics toolbox
    event_data = importdata(events_file);
    event_data = cellfun(@strsplit, event_data(2:end), 'UniformOutput', false);
    event_data = cellfun(@(x)x(6), event_data);
    nodes = cellfun(@str2num, event_data);

    for eventidx = 2:60
        zfile = fullfile(lss_path, sprintf('run-%d/zfiles/fwhm-5.0_event-%d_zstat1.nii.gz', runidx, eventidx - 1));
        ds = cosmo_fmri_dataset(zfile, ...
                                'mask', bold_mask_path, ...
                                'targets', nodes(eventidx), ...
                                'chunks', runidx);
        counter = counter + 1;
        ds_lss_zfiles_all{counter} = ds;
    end
end
[~, ds_int_cell] = cosmo_mask_dim_intersect(ds_lss_zfiles_all);
ds_lss_zfiles=cosmo_stack(ds_int_cell,1);
clear ds_lss_zfiles_all idx_cell ds_int_cell;
% remove constant features (due to liberal masking)
ds_lss_zfiles=cosmo_remove_useless_data(ds_lss_zfiles);
% Average patterns within runs
ds_lss_zfiles_avg = cosmo_fx(ds_lss_zfiles, @(x)(mean(x,1)), {'targets'}, 1);
% Zscore within chunks
splits = cosmo_split(ds_lss_zfiles, 'chunks');
nsplits=numel(splits);
% allocate space for output
outputs=cell(nsplits,1);
for k=1:nsplits
    d=splits{k};
    d.samples = d.samples - mean(d.samples);
    d.samples = d.samples ./ std(d.samples);
    outputs{k} = d;
end
ds_lss_zfiles_zscore = cosmo_stack(outputs);
ds_lss_zfiles_zscore_avg = cosmo_fx(ds_lss_zfiles_zscore, @(x)(mean(x,1)), {'targets'}, 1);
clear ds_lss_zfiles_zscore ds_lss_zfiles;
%% Searchlight Neighborhood
disp('Creating neighborhood');
% Use a searchlight with a more-or-less constant number of voxels,
% both near the edges of the brain and in the center of the brain.
nvoxels_per_searchlight=100;

% Define a spherical neighborhood with approximately
% 100 voxels around each voxel using cosmo_spherical_neighborhood,
% and assign the result to a variable named 'nbrhood'
%     nbrhood_lss_zfiles = cosmo_spherical_neighborhood(ds_lss_zfiles_avg, 'count', nvoxels_per_searchlight);
% nbrhood_lsa_zfiles = cosmo_spherical_neighborhood(ds_lsa_zfiles_avg, 'count', nvoxels_per_searchlight);
% nbrhood_lss_pes = cosmo_spherical_neighborhood(ds_lss_pes_avg, 'count', nvoxels_per_searchlight);
% nbrhood_lsa_pes = cosmo_spherical_neighborhood(ds_lsa_pes_avg, 'count', nvoxels_per_searchlight);
nbrhood_lss_zfiles_zscore = cosmo_spherical_neighborhood(ds_lss_zfiles_zscore_avg, 'count', nvoxels_per_searchlight);
% nbrhood_lsa_zfiles_zscore = cosmo_spherical_neighborhood(ds_lsa_zfiles_zscore_avg, 'count', nvoxels_per_searchlight);
% nbrhood_lss_pes_zscore = cosmo_spherical_neighborhood(ds_lss_pes_zscore_avg, 'count', nvoxels_per_searchlight);
% nbrhood_lsa_pes_zscore = cosmo_spherical_neighborhood(ds_lsa_pes_zscore_avg, 'count', nvoxels_per_searchlight);
%% Crossvalidation Measure
% define the measure as cosmo_crossvalidation_measure
% measure = @cosmo_crossvalidation_measure;
measure = @cosmo_dissimilarity_matrix_measure;

% Define measure arguments.
% Ideally we would like to use nfold_partitioner (and you probably
% want to do this for a publication-quality), but this takes quite long.
% Instead, we take a short-cut here and use an
% odd-even partitioning scheme that has only one fold (test on odd chunks,
% test on even) using cosmo_oddeven_partitioner.

measure_args=struct();
% Enable centering the data
measure_args.center_data=true;
%% Run SVM
% Using the neighborhood, the measure, and the measure's arguments, run the
% searchlight using cosmo_searchlight. Assign the result to a variable
% named 'ds_cfy'
disp('Running searchlight');
ds_cf_lss_zfiles = cosmo_searchlight(ds_lss_zfiles_avg, nbrhood_lss_zfiles_zscore, measure, measure_args, 'nproc', nproc);
ds_cf_lss_zfiles_zscored = cosmo_searchlight(ds_lss_zfiles_zscore_avg, nbrhood_lss_zfiles_zscore, measure, measure_args, 'nproc', nproc);

%% Save results
output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm_nvox-%.0f_searchlight-lss-zfiles.nii.gz', subject, nvoxels_per_searchlight));
cosmo_map2fmri(ds_cf_lss_zfiles, output_fn);

output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm_nvox-%.0f_searchlight-lss-zfiles-zscored.nii.gz', subject, nvoxels_per_searchlight));
cosmo_map2fmri(ds_cf_lss_zfiles_zscored, output_fn);
end
