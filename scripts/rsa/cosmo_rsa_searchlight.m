function exitcode = cosmo_rsa_searchlight(subject, project_path, output_path, nproc)
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
nevents = 59;

% subjects = {'GLS003', 'GLS004', 'GLS005', 'GLS006', ...
%             'GLS008', 'GLS009', 'GLS010', 'GLS011', ...
%             'GLS013', 'GLS014', 'GLS017', 'GLS018', ...
%             'GLS019', 'GLS020', 'GLS021', 'GLS022', ...
%             'GLS023', 'GLS024', 'GLS025', 'GLS026', ...
%             'GLS027', 'GLS028', 'GLS030', 'GLS033', ...
%             'GLS037', 'GLS038', 'GLS039', 'GLS040', ...
%             'GLS043', 'GLS044', 'GLS045'};

% RDM

lattice_rdm = dlmread('dissim_lattice.txt');
lattice_rdm = lattice_rdm - diag(diag(lattice_rdm));
modular_rdm = dlmread('dissim_modular.txt');
modular_rdm = modular_rdm - diag(diag(modular_rdm));

%% Paths
fmriprep_func_path = fullfile(project_path, sprintf('derived/fmriprep/sub-%s/ses-2/func', subject));
event_path = fullfile(project_path, sprintf('data/sub-%s/ses-2/func', subject));
lss_path = fullfile(project_path, sprintf('derived/feat_representation_lss/sub-%s', subject));
lsa_path = fullfile(project_path, sprintf('derived/feat_representation_lsa/sub-%s', subject));
%% Combined mask
disp('Making combined mask');
mask1 = load_nii(fullfile(fmriprep_func_path, sprintf('sub-%s_ses-2_task-graphrepresentation_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz', subject)));
mask2 = load_nii(fullfile(fmriprep_func_path, sprintf('sub-%s_ses-2_task-graphrepresentation_run-2_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz', subject)));
mask3 = load_nii(fullfile(fmriprep_func_path, sprintf('sub-%s_ses-2_task-graphrepresentation_run-3_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz', subject)));
mask4 = load_nii(fullfile(fmriprep_func_path, sprintf('sub-%s_ses-2_task-graphrepresentation_run-4_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz', subject)));
mask5 = load_nii(fullfile(fmriprep_func_path, sprintf('sub-%s_ses-2_task-graphrepresentation_run-5_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz', subject)));
mask6 = load_nii(fullfile(fmriprep_func_path, sprintf('sub-%s_ses-2_task-graphrepresentation_run-6_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz', subject)));
mask7 = load_nii(fullfile(fmriprep_func_path, sprintf('sub-%s_ses-2_task-graphrepresentation_run-7_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz', subject)));
mask8 = load_nii(fullfile(fmriprep_func_path, sprintf('sub-%s_ses-2_task-graphrepresentation_run-8_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz', subject)));
bold_mask = mask1;
bold_mask.img = mask1.img & mask2.img & mask3.img & mask4.img & mask5.img & mask6.img & mask7.img & mask8.img;
bold_mask_path = fullfile(output_path, sprintf('masks/sub-%s_desc-boldmask_combined.nii.gz', subject));
save_nii(bold_mask, bold_mask_path);

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
    d.samples = d.samples / std(d.samples);
    outputs{k} = d;
end
ds_lss_zfiles_zscore = cosmo_stack(splits);
ds_lss_zfiles_zscore_avg = cosmo_fx(ds_lss_zfiles_zscore, @(x)(mean(x,1)), {'targets'}, 1);
clear ds_lss_zfiles_zscore ds_lss_zfiles splits outputs;

disp('pes');
ds_lss_pes_all = cell(59 * nruns, 1);
counter = 0;
for runidx = 1:nruns
    events_file = fullfile(event_path, sprintf('sub-%s_ses-2_task-graphrepresentation_run-%d_events.tsv', subject, runidx));

    % avoid statistics toolbox
    event_data = importdata(events_file);
    event_data = cellfun(@strsplit, event_data(2:end), 'UniformOutput', false);
    event_data = cellfun(@(x)x(6), event_data);
    nodes = cellfun(@str2num, event_data);

    for eventidx = 2:60
        zfile = fullfile(lss_path, sprintf('run-%d/parameter_estimates/fwhm-5.0_event-%d_pe1.nii.gz', runidx, eventidx - 1));
        ds = cosmo_fmri_dataset(zfile, ...
                                'mask', bold_mask_path, ...
                                'targets', nodes(eventidx), ...
                                'chunks', runidx);
        counter = counter + 1;
        ds_lss_pes_all{counter} = ds;
    end
end
[~, ds_int_cell] = cosmo_mask_dim_intersect(ds_lss_pes_all);
ds_lss_pes=cosmo_stack(ds_int_cell,1);
clear ds_lss_pes_all idx_cell ds_int_cell;
% remove constant features (due to liberal masking)
ds_lss_pes=cosmo_remove_useless_data(ds_lss_pes);
ds_lss_pes_avg = cosmo_fx(ds_lss_pes, @(x)(mean(x,1)), {'targets'}, 1);
splits = cosmo_split(ds_lss_pes, 'chunks');
nsplits=numel(splits);
% allocate space for output
outputs=cell(nsplits,1);
for k=1:nsplits
    d=splits{k};
    d.samples = d.samples - mean(d.samples);
    d.samples = d.samples / std(d.samples);
    outputs{k} = d;
end
ds_lss_pes_zscore = cosmo_stack(splits);
ds_lss_pes_zscore_avg = cosmo_fx(ds_lss_pes_zscore, @(x)(mean(x,1)), {'targets'}, 1);
clear ds_lss_pes_zscore ds_lss_pes splits outputs;


disp('Loading LS-A data');
disp('zfiles');
ds_lsa_zfiles_all = cell(15 * nruns, 1);
counter = 0;
for runidx = 1:nruns
    for nodeidx = 1:15
        zfile = fullfile(lsa_path, sprintf('run-%d/zfiles/fwhm-5.0_zstat%d.nii.gz', runidx, nodeidx));
        ds = cosmo_fmri_dataset(zfile, ...
                                'mask', bold_mask_path, ...
                                'targets', nodeidx, ...
                                'chunks', runidx);
        counter = counter + 1;
        ds_lsa_zfiles_all{counter} = ds;
    end
end
[~, ds_int_cell] = cosmo_mask_dim_intersect(ds_lsa_zfiles_all);
ds_lsa_zfiles=cosmo_stack(ds_int_cell,1);
clear ds_lsa_zfiles_all idx_cell ds_int_cell;
% remove constant features (due to liberal masking)
ds_lsa_zfiles=cosmo_remove_useless_data(ds_lsa_zfiles);
ds_lsa_zfiles_avg = cosmo_fx(ds_lsa_zfiles, @(x)(mean(x,1)), {'targets'}, 1);
splits = cosmo_split(ds_lsa_zfiles, 'chunks');
nsplits=numel(splits);
% allocate space for output
outputs=cell(nsplits,1);
for k=1:nsplits
    d=splits{k};
    d.samples = d.samples - mean(d.samples);
    d.samples = d.samples / std(d.samples);
    outputs{k} = d;
end
ds_lsa_zfiles_zscore = cosmo_stack(splits);
ds_lsa_zfiles_zscore_avg = cosmo_fx(ds_lsa_zfiles_zscore, @(x)(mean(x,1)), {'targets'}, 1);
clear ds_lsa_zfiles_zscore ds_lsa_zfiles splits outputs;

disp('pes');
ds_lsa_pes_all = cell(15 * nruns, 1);
counter = 0;
for runidx = 1:nruns
    for nodeidx = 1:15
        zfile = fullfile(lsa_path, sprintf('run-%d/parameter_estimates/fwhm-5.0_pe%d.nii.gz', runidx, nodeidx));
        ds = cosmo_fmri_dataset(zfile, ...
                                'mask', bold_mask_path, ...
                                'targets', nodeidx, ...
                                'chunks', runidx);
        counter = counter + 1;
        ds_lsa_pes_all{counter} = ds;
    end
end
[~, ds_int_cell] = cosmo_mask_dim_intersect(ds_lsa_pes_all);
ds_lsa_pes=cosmo_stack(ds_int_cell,1);
clear ds_lsa_pes_all idx_cell ds_int_cell;
% remove constant features (due to liberal masking)
ds_lsa_pes=cosmo_remove_useless_data(ds_lsa_pes);
ds_lsa_pes_avg = cosmo_fx(ds_lsa_pes, @(x)(mean(x,1)), {'targets'}, 1);
splits = cosmo_split(ds_lsa_pes, 'chunks');
nsplits=numel(splits);
% allocate space for output
outputs=cell(nsplits,1);
for k=1:nsplits
    d=splits{k};
    d.samples = d.samples - mean(d.samples);
    d.samples = d.samples / std(d.samples);
    outputs{k} = d;
end
ds_lsa_pes_zscore = cosmo_stack(splits);
ds_lsa_pes_zscore_avg = cosmo_fx(ds_lsa_pes_zscore, @(x)(mean(x,1)), {'targets'}, 1);
clear ds_lsa_pes_zscore ds_lsa_pes splits outputs;
%% Searchlight Neighborhood
disp('Creating neighborhood');
% Use a searchlight with a more-or-less constant number of voxels,
% both near the edges of the brain and in the center of the brain.
nvoxels_values = {50, 100, 200};
for nvoxel_id = 1:length(nvoxels_values)

    nvoxels_per_searchlight = nvoxels_values{nvoxel_id};

    % Define a spherical neighborhood with approximately
    % 100 voxels around each voxel using cosmo_spherical_neighborhood,
    % and assign the result to a variable named 'nbrhood'
    nbrhood_lss_zfiles = cosmo_spherical_neighborhood(ds_lss_zfiles_avg, 'count', nvoxels_per_searchlight);
    nbrhood_lsa_zfiles = cosmo_spherical_neighborhood(ds_lsa_zfiles_avg, 'count', nvoxels_per_searchlight);
    nbrhood_lss_pes = cosmo_spherical_neighborhood(ds_lss_pes_avg, 'count', nvoxels_per_searchlight);
    nbrhood_lsa_pes = cosmo_spherical_neighborhood(ds_lsa_pes_avg, 'count', nvoxels_per_searchlight);
    nbrhood_lss_zfiles_zscore = cosmo_spherical_neighborhood(ds_lss_zfiles_zscore_avg, 'count', nvoxels_per_searchlight);
    nbrhood_lsa_zfiles_zscore = cosmo_spherical_neighborhood(ds_lsa_zfiles_zscore_avg, 'count', nvoxels_per_searchlight);
    nbrhood_lss_pes_zscore = cosmo_spherical_neighborhood(ds_lss_pes_zscore_avg, 'count', nvoxels_per_searchlight);
    nbrhood_lsa_pes_zscore = cosmo_spherical_neighborhood(ds_lsa_pes_zscore_avg, 'count', nvoxels_per_searchlight);

    metrics = {'Kendall', 'Pearson'};
    for metric_id = 1:length(metrics)
        metric = metrics{metric_id};
        %% Crossvalidation Measure
        % define the measure as cosmo_crossvalidation_measure
        % measure = @cosmo_crossvalidation_measure;
        measure = @cosmo_target_dsm_corr_measure;

        % Define measure arguments.
        % Ideally we would like to use nfold_partitioner (and you probably
        % want to do this for a publication-quality), but this takes quite long.
        % Instead, we take a short-cut here and use an
        % odd-even partitioning scheme that has only one fold (test on odd chunks,
        % test on even) using cosmo_oddeven_partitioner.

        measure_args_lattice=struct();
        measure_args_lattice.target_dsm = lattice_rdm;
        % Enable centering the data
        measure_args_lattice.center_data=true;
        measure_args_lattice.type = metric;

        measure_args_modular=struct();
        measure_args_modular.target_dsm = modular_rdm;
        % Enable centering the data
        measure_args_modular.center_data=true;
        measure_args_modular.type = metric;
        %% Run SVM
        % Using the neighborhood, the measure, and the measure's arguments, run the
        % searchlight using cosmo_searchlight. Assign the result to a variable
        % named 'ds_cfy'
        disp('Running searchlight');
        ds_cfy_lattice_lss_zfiles = cosmo_searchlight(ds_lss_zfiles_avg, nbrhood_lss_zfiles, measure, measure_args_lattice, 'nproc', nproc);
        ds_cfy_lattice_lsa_zfiles = cosmo_searchlight(ds_lsa_zfiles_avg, nbrhood_lsa_zfiles, measure, measure_args_lattice, 'nproc', nproc);
        ds_cfy_lattice_lss_zfiles_zscore = cosmo_searchlight(ds_lss_zfiles_zscore_avg, nbrhood_lss_zfiles_zscore, measure, measure_args_lattice, 'nproc', nproc);
        ds_cfy_lattice_lsa_zfiles_zscore = cosmo_searchlight(ds_lsa_zfiles_zscore_avg, nbrhood_lsa_zfiles_zscore, measure, measure_args_lattice, 'nproc', nproc);
        ds_cfy_lattice_lss_pes = cosmo_searchlight(ds_lss_pes_avg, nbrhood_lss_pes, measure, measure_args_lattice, 'nproc', nproc);
        ds_cfy_lattice_lsa_pes = cosmo_searchlight(ds_lsa_pes_avg, nbrhood_lsa_pes, measure, measure_args_lattice, 'nproc', nproc);
        ds_cfy_lattice_lss_pes_zscore = cosmo_searchlight(ds_lss_pes_zscore_avg, nbrhood_lss_pes_zscore, measure, measure_args_lattice, 'nproc', nproc);
        ds_cfy_lattice_lsa_pes_zscore = cosmo_searchlight(ds_lsa_pes_zscore_avg, nbrhood_lsa_pes_zscore, measure, measure_args_lattice, 'nproc', nproc);
        ds_cfy_modular_lss_zfiles = cosmo_searchlight(ds_lss_zfiles_avg, nbrhood_lss_zfiles, measure, measure_args_modular, 'nproc', nproc);
        ds_cfy_modular_lsa_zfiles = cosmo_searchlight(ds_lsa_zfiles_avg, nbrhood_lsa_zfiles, measure, measure_args_modular, 'nproc', nproc);
        ds_cfy_modular_lss_zfiles_zscore = cosmo_searchlight(ds_lss_zfiles_zscore_avg, nbrhood_lss_zfiles_zscore, measure, measure_args_modular, 'nproc', nproc);
        ds_cfy_modular_lsa_zfiles_zscore = cosmo_searchlight(ds_lsa_zfiles_zscore_avg, nbrhood_lsa_zfiles_zscore, measure, measure_args_modular, 'nproc', nproc);
        ds_cfy_modular_lss_pes = cosmo_searchlight(ds_lss_pes_avg, nbrhood_lss_pes, measure, measure_args_modular, 'nproc', nproc);
        ds_cfy_modular_lsa_pes = cosmo_searchlight(ds_lsa_pes_avg, nbrhood_lsa_pes, measure, measure_args_modular, 'nproc', nproc);
        ds_cfy_modular_lss_pes_zscore = cosmo_searchlight(ds_lss_pes_zscore_avg, nbrhood_lss_pes_zscore, measure, measure_args_modular, 'nproc', nproc);
        ds_cfy_modular_lsa_pes_zscore = cosmo_searchlight(ds_lsa_pes_zscore_avg, nbrhood_lsa_pes_zscore, measure, measure_args_modular, 'nproc', nproc);
        %% Save results
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-lattice_metric-%s_nvox-%.0f_searchlight-lss-zfiles.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_lattice_lss_zfiles, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-lattice_metric-%s_nvox-%.0f_searchlight-lsa-zfiles.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_lattice_lsa_zfiles, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-lattice_metric-%s_nvox-%.0f_searchlight-lss-zfiles-zscore.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_lattice_lss_zfiles_zscore, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-lattice_metric-%s_nvox-%.0f_searchlight-lsa-zfiles-zscore.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_lattice_lsa_zfiles_zscore, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-lattice_metric-%s_nvox-%.0f_searchlight-lss-pes.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_lattice_lss_pes, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-lattice_metric-%s_nvox-%.0f_searchlight-lsa-pes.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_lattice_lsa_pes, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-lattice_metric-%s_nvox-%.0f_searchlight-lss-pes-zscore.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_lattice_lss_pes_zscore, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-lattice_metric-%s_nvox-%.0f_searchlight-lsa-pes-zscore.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_lattice_lsa_pes_zscore, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-modular_metric-%s_nvox-%.0f_searchlight-lss-zfiles.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_modular_lss_zfiles, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-modular_metric-%s_nvox-%.0f_searchlight-lsa-zfiles.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_modular_lsa_zfiles, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-modular_metric-%s_nvox-%.0f_searchlight-lss-zfiles-zscore.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_modular_lss_zfiles_zscore, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-modular_metric-%s_nvox-%.0f_searchlight-lsa-zfiles-zscore.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_modular_lsa_zfiles_zscore, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-modular_metric-%s_nvox-%.0f_searchlight-lss-pes.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_modular_lss_pes, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-modular_metric-%s_nvox-%.0f_searchlight-lsa-pes.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_modular_lsa_pes, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-modular_metric-%s_nvox-%.0f_searchlight-lss-pes-zscore.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_modular_lss_pes_zscore, output_fn);
        output_fn=fullfile(output_path, 'results', sprintf('sub-%s_rdm-modular_metric-%s_nvox-%.0f_searchlight-lsa-pes-zscore.nii.gz', subject, metric, nvoxels_per_searchlight));
        cosmo_map2fmri(ds_cfy_modular_lsa_pes_zscore, output_fn);
    end
    %% Crossvalidation Measure
    % define the measure as cosmo_crossvalidation_measure
    % measure = @cosmo_crossvalidation_measure;
    measure_glm = @cosmo_target_dsm_corr_measure;
    % Define measure arguments.
    % Ideally we would like to use nfold_partitioner (and you probably
    % want to do this for a publication-quality), but this takes quite long.
    % Instead, we take a short-cut here and use an
    % odd-even partitioning scheme that has only one fold (test on odd chunks,
    % test on even) using cosmo_oddeven_partitioner.

    measure_args_glm=struct();

    measure_args_glm.glm_dsm = {lattice_rdm, modular_rdm};
    % Enable centering the data
    measure_args_glm.center_data=true;
    % measure_args_glm.type = 'Kendall';

    disp('Running searchlight GLM');
    ds_cfy_glm_lss_zfiles = cosmo_searchlight(ds_lss_zfiles_avg, nbrhood_lss_zfiles, measure_glm, measure_args_glm, 'nproc', nproc);
    ds_cfy_glm_lsa_zfiles = cosmo_searchlight(ds_lsa_zfiles_avg, nbrhood_lsa_zfiles, measure_glm, measure_args_glm, 'nproc', nproc);
    ds_cfy_glm_lss_zfiles_zscore = cosmo_searchlight(ds_lss_zfiles_zscore_avg, nbrhood_lss_zfiles_zscore, measure_glm, measure_args_glm, 'nproc', nproc);
    ds_cfy_glm_lsa_zfiles_zscore = cosmo_searchlight(ds_lsa_zfiles_zscore_avg, nbrhood_lsa_zfiles_zscore, measure_glm, measure_args_glm, 'nproc', nproc);
    ds_cfy_glm_lss_pes = cosmo_searchlight(ds_lss_pes_avg, nbrhood_lss_pes, measure_glm, measure_args_glm, 'nproc', nproc);
    ds_cfy_glm_lsa_pes = cosmo_searchlight(ds_lsa_pes_avg, nbrhood_lsa_pes, measure_glm, measure_args_glm, 'nproc', nproc);
    ds_cfy_glm_lss_pes_zscore = cosmo_searchlight(ds_lss_pes_zscore_avg, nbrhood_lss_pes_zscore, measure_glm, measure_args_glm, 'nproc', nproc);
    ds_cfy_glm_lsa_pes_zscore = cosmo_searchlight(ds_lsa_pes_zscore_avg, nbrhood_lsa_pes_zscore, measure_glm, measure_args_glm, 'nproc', nproc);
    %% Save
    % Set output filename
    output_fn=fullfile(output_path, 'results', ...
        sprintf('sub-%s_rdm-glm-lattice-modular_nvox-%.0f_searchlight-lss-zfiles.nii.gz', subject, nvoxels_per_searchlight));
    cosmo_map2fmri(ds_cfy_glm_lss_zfiles, output_fn);
    output_fn=fullfile(output_path, 'results', ...
        sprintf('sub-%s_rdm-glm-lattice-modular_nvox-%.0f_searchlight-lsa-zfiles.nii.gz', subject, nvoxels_per_searchlight));
    cosmo_map2fmri(ds_cfy_glm_lsa_zfiles, output_fn);
    output_fn=fullfile(output_path, 'results', ...
        sprintf('sub-%s_rdm-glm-lattice-modular_nvox-%.0f_searchlight-lss-zfiles-zscore.nii.gz', subject, nvoxels_per_searchlight));
    cosmo_map2fmri(ds_cfy_glm_lss_zfiles_zscore, output_fn);
    output_fn=fullfile(output_path, 'results', ...
        sprintf('sub-%s_rdm-glm-lattice-modular_nvox-%.0f_searchlight-lsa-zfiles-zscore.nii.gz', subject, nvoxels_per_searchlight));
    cosmo_map2fmri(ds_cfy_glm_lsa_zfiles_zscore, output_fn);
    output_fn=fullfile(output_path, 'results', ...
        sprintf('sub-%s_rdm-glm-lattice-modular_nvox-%.0f_searchlight-lss-pes.nii.gz', subject, nvoxels_per_searchlight));
    cosmo_map2fmri(ds_cfy_glm_lss_pes, output_fn);
    output_fn=fullfile(output_path, 'results', ...
        sprintf('sub-%s_rdm-glm-lattice-modular_nvox-%.0f_searchlight-lsa-pes.nii.gz', subject, nvoxels_per_searchlight));
    cosmo_map2fmri(ds_cfy_glm_lsa_pes, output_fn);
    output_fn=fullfile(output_path, 'results', ...
        sprintf('sub-%s_rdm-glm-lattice-modular_nvox-%.0f_searchlight-lss-pes-zscore.nii.gz', subject, nvoxels_per_searchlight));
    cosmo_map2fmri(ds_cfy_glm_lss_pes_zscore, output_fn);
    output_fn=fullfile(output_path, 'results', ...
        sprintf('sub-%s_rdm-glm-lattice-modular_nvox-%.0f_searchlight-lsa-pes-zscore.nii.gz', subject, nvoxels_per_searchlight));
    cosmo_map2fmri(ds_cfy_glm_lsa_pes_zscore, output_fn);
    end
end
