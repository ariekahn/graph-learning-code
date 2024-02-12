#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
=========================
fMRI: FSL localizer workflow
=========================
"""

import os  # system functions
import argparse
import nipype.interfaces.fsl as fsl  # fsl
import nipype.algorithms.modelgen as model  # model generation
# import nipype.algorithms.rapidart as ra  # artifact detection
from nipype import SelectFiles, Node, Workflow, IdentityInterface, DataSink, Function

from graphlearning_mvpa.preprocess import create_fsl_fmriprep_preproc
from graphlearning_mvpa.estimate import create_modelfit_workflow, create_postprocessing_workflow
from graphlearning_mvpa.util import ExtractConfounds, FindNonDummyTime

parser = argparse.ArgumentParser('Run FSL localizer')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('project_dir', type=str, help='directory containing data and derived subfolders')
parser.add_argument('work_dir', type=str, help='Temporary work directory')
parser.add_argument('output_dir', type=str, help='Location for output files')
parser.add_argument('--dry_run', action='store_true', help="Attempt to create workflow but don't run.")
parser.add_argument('--debug', action='store_true', help="Run nipype in debug mode")
parser.add_argument('--sge', action='store_true', help="Run using SGE")
args = parser.parse_args()

if args.debug:
    print('Running in debug mode')
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)

"""
Preliminaries
-------------

Setup any package specific configuration. The output file format for FSL
routines is being set to compressed NIFTI.
"""

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

"""
First-Level Workflow
--------------------

Create our workflow.

The main components of the workflow are
1. preproc, which performs smoothing (FSL SUSAN) and highpass-filtering
2. modelspec, which generates a design matrix
3. modelfit, which fits a GLM to the preprocessed data
"""

level1_workflow = Workflow(name='feat_localizer')
level1_workflow.base_dir = os.path.abspath(args.work_dir)
level1_workflow.config['execution'] = dict(
    crashdump_dir=os.path.abspath(f'{args.work_dir}/crashdumps'))

preproc = create_fsl_fmriprep_preproc(whichvol='first')

modelfit = create_modelfit_workflow()

modelspec = Node(model.SpecifyModel(), name="modelspec")

postprocessing = create_postprocessing_workflow()

level1_workflow.connect([
    (preproc, modelspec, [('outputspec.highpassed_files', 'functional_runs')]),
    (modelspec, modelfit, [('session_info', 'inputspec.session_info')]),
    (preproc, modelfit, [('outputspec.highpassed_files',
                          'inputspec.functional_data')])
])

# def sort_copes(files):
#     numelements = len(files[0])
#     outfiles = []
#     for i in range(numelements):
#         outfiles.insert(i, [])
#         for j, elements in enumerate(files):
#             outfiles[i].append(elements[i])
#     return outfiles


# def num_copes(files):
#     return len(files)


# pickfirst = lambda x: x[0]

# level1_workflow.connect(
#     [(preproc, fixed_fx, [(('outputspec.mask', pickfirst),
#                            'flameo.mask_file')]),
#      (modelfit, fixed_fx, [
#          (('outputspec.copes', sort_copes), 'inputspec.copes'),
#          ('outputspec.dof_file', 'inputspec.dof_files'),
#          (('outputspec.varcopes', sort_copes), 'inputspec.varcopes'),
#          (('outputspec.copes', num_copes), 'l2model.num_copes'),
#      ])])
"""
Experiment specific components
------------------------------

"""

# Specify the subject directories
subject_list = [args.subject_id]

infosource = Node(
    IdentityInterface(fields=['subject_id']), name="infosource")
infosource.iterables = ('subject_id', subject_list)

# String template with {}-based strings
templates = {
    'func': 'derived/fmriprep/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-{space}_desc-preproc_bold.nii.gz',
    'mask': 'derived/fmriprep/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-{space}_desc-brain_mask.nii.gz',
    'confounds': 'derived/fmriprep/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_desc-confounds_regressors.tsv',
    'events': 'data/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_events.tsv',
}
# Create SelectFiles node
sf = Node(SelectFiles(templates),
          name='selectfiles')
# Location of the dataset folder
sf.inputs.base_directory = os.path.abspath(args.project_dir)
# Feed {}-based placeholder strings with values
sf.inputs.space = 'MNI152NLin2009cAsym'

"""
Set up the contrast structure that needs to be evaluated. This is a list of
lists. The inner list specifies the contrasts and has the following format -
[Name,Stat,[list of condition names],[weights on those conditions]. The
condition names must match the `names` listed in the `subjectinfo` function
described above.
"""

cont1 = ['Object>Random', 'T', ['object', 'random'], [1, -1]]
cont2 = ['Visual', 'T', ['object', 'random'], [0.5, 0.5]]
contrasts = [cont1, cont2]

"""
Specify parameters for running our design matrix
First, wee need the high-pass cutoff and TR
"""
hpcutoff = 72.0  # Entire cycle of 3 runs is 36 seconds
TR = 0.8

modelspec.inputs.input_units = 'secs'
modelspec.inputs.time_repetition = TR
modelspec.inputs.high_pass_filter_cutoff = hpcutoff

"""
Next, we specify our HRF model (double-gamma), and our contrast
"""

modelfit.inputs.inputspec.interscan_interval = TR
modelfit.inputs.inputspec.bases = {'dgamma': {'derivs': True}}
modelfit.inputs.inputspec.contrasts = contrasts
modelfit.inputs.inputspec.model_serial_correlations = True
modelfit.inputs.inputspec.film_threshold = 1000

"""
Events and Confounds

Confounds: We need to to extract the fmriprep confounds into a format
that FSL can use. This includes motion parameters, as well as high-motion
volumes. We use a workflow (extractconfounds) to do this.

We can directly read events out of the tsv file in the original BIDS dir.

We'll hook these up to the main workflow.
"""

extractconfounds = Node(ExtractConfounds(), name="extractconfounds")
# motion_6 or motion_24
extractconfounds.inputs.motion_params = 'motion_24'


def get_localizer_events(event_file, non_dummy_time):
    from nipype.interfaces.base import Bunch
    import pandas as pd
    events = pd.read_csv(event_file, sep='\t')

    # Filter by valid start times
    events = events[events.onset >= non_dummy_time]

    onsets = []
    durations = []
    conditions = ['object', 'random']
    for c in conditions:
        onsets.append(events[events['trial_type'] == c]['onset'])
        durations.append(events[events['trial_type'] == c]['duration'])
    output = Bunch(
        conditions=conditions,
        onsets=onsets,
        durations=durations
    )
    return output


findNonDummyTime = Node(Function(input_names=['regressors_file', 'tr'],
                                 output_names=['non_dummy_time'],
                                 function=FindNonDummyTime),
                        name='findNonDummyTime')
findNonDummyTime.inputs.tr = TR

getevents = Node(Function(input_names=['event_file', 'non_dummy_time'],
                          output_names=['events'],
                          function=get_localizer_events),
                 name='getevents')

level1_workflow.connect([
    (infosource, sf, [('subject_id', 'subject_id')]),
    # Event parsing
    (sf, findNonDummyTime, [('confounds', 'regressors_file')]),
    (sf, getevents, [('events', 'event_file')]),
    (findNonDummyTime, getevents, [('non_dummy_time', 'non_dummy_time')]),
    (getevents, modelspec, [('events', 'subject_info')]),

    (sf, preproc, [('func', 'inputspec.func'),
                   ('mask', 'inputspec.mask'),
                   ]),
    (sf, extractconfounds, [('confounds', 'in_file')]),
    (extractconfounds, modelspec, [('out_file', 'realignment_parameters')]),
])

"""
Postprocessing workflow (Thresholded Clusters)
"""

level1_workflow.connect([
    (sf, postprocessing, [('mask', 'inputspec.mask')]),
    (modelfit, postprocessing, [('outputspec.dof_file', 'inputspec.dof_file'),
                                ('modelestimate.residual4d', 'inputspec.res4d'),
                                ('outputspec.zfiles', 'inputspec.zfiles')]),
])
"""
Smoothing

Use the get_node function to retrieve an internal node by name. Then set the
iterables on this node to perform two different extents of smoothing.
"""

featinput = level1_workflow.get_node('preproc.inputspec')
fwhm = [5.]
featinput.iterables = ('fwhm', fwhm)

featinput.inputs.highpass = hpcutoff / (2. * TR)

"""
Output
"""
# Create DataSink object
sink = Node(DataSink(), name='sink')

# Name of the output folder
sink.inputs.base_directory = os.path.abspath(args.output_dir)

substitutions = [('_event_id_', ''),
                 ('_run_', 'run-'),
                 ('_mcf', ''),
                 ('_st', ''),
                 ('_flirt', ''),
                 ('.nii_mean_reg', '_mean'),
                 ('_modelestimate0/', ''),
                 ('.nii.par', '.par'),
                 ('_maths_index.nii.gz', '_index.nii.gz'),
                 ('_maths.txt', '.txt'),
                 ('_maths_threshold.nii.gz', '_threshold.nii.gz'),
                 ]
substitutions += [('_subject_id_%s/' % s, '') for s in subject_list]
for i in range(len(contrasts)):
    substitutions += [(f'_ztop{i}/', '')]
for f in fwhm:
    substitutions += [(f'_fwhm_{f}/', f'fwhm-{f}_')]
sink.inputs.substitutions = substitutions

regexp_substitutions = [(r'_cluster[0-9]*\/', '')]

sink.inputs.regexp_substitutions = regexp_substitutions

# Connect DataSink with the relevant nodes
level1_workflow.connect([
    (modelfit, sink, [
        ('outputspec.copes', 'copes'),
        ('outputspec.varcopes', 'varcopes'),
        ('outputspec.dof_file', 'dof_file'),
        ('outputspec.pfiles', 'pfiles'),
        ('outputspec.zfiles', 'zfiles'),
        ('outputspec.parameter_estimates', 'parameter_estimates')]),
    (postprocessing, sink, [
        ('outputspec.index_file', 'clusters.@index_file'),
        ('outputspec.threshold_file', 'clusters.@threshold_file'),
        ('outputspec.cluster_output', 'clusters.@cluster_output'),
    ]),
])

"""
Execute the pipeline
--------------------

The code discussed above sets up all the necessary data structures with
appropriate parameters and the connectivity between the processes, but does not
generate any output. To actually run the analysis on the data the
``nipype.pipeline.engine.Pipeline.Run`` function needs to be called.
"""

level1_workflow.write_graph(graph2use='exec')
if not args.dry_run:
    if args.sge:
        level1_workflow.run(plugin='SGEGraph',
                            plugin_args={
                                'dont_resubmit_completed_jobs': True,
                                'qsub_args': '-l h_vmem=10G,s_vmem=9.5G -j y',
                            })
    else:
        level1_workflow.run()
