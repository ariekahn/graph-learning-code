#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
=========================
fMRI: FSL reuse workflows
=========================

A workflow that uses fsl to perform a first level analysis on the nipype
tutorial data set::

    python fmri_fsl_reuse.py


First tell python where to find the appropriate functions.
"""

import os  # system functions
import argparse
import nipype.interfaces.fsl as fsl  # fsl
import nipype.algorithms.modelgen as model  # model generation
# import nipype.algorithms.rapidart as ra  # artifact detection
from nipype import SelectFiles, Node, Workflow, IdentityInterface, DataSink, Function

from graphlearning_mvpa.preprocess import create_fsl_fmriprep_preproc
from graphlearning_mvpa.estimate import create_modelfit_workflow
from graphlearning_mvpa.util import ExtractConfounds, FindNonDummyTime

parser = argparse.ArgumentParser('Run FSL Representation analysis in LS-S style')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('run', type=int, help='Run to process')
parser.add_argument('project_dir', type=str, help='directory containing data and derived subfolders')
parser.add_argument('work_dir', type=str, help='Temporary work directory')
parser.add_argument('output_dir', type=str, help='Location for output files')
parser.add_argument('--dry_run', action='store_true', help="Attempt to create workflow but don't run.")
parser.add_argument('--debug', action='store_true', help="Run nipype in debug mode")
cluster_opts = parser.add_mutually_exclusive_group()
cluster_opts.add_argument('--sge', action='store_true', help="Run using SGE")
cluster_opts.add_argument('--n_procs', type=int, help="Number of processers to use")
args = parser.parse_args()

if args.debug:
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

level1_workflow = Workflow(name='feat_representation_lss')
level1_workflow.base_dir = os.path.abspath(args.work_dir)
level1_workflow.config['execution'] = dict(
    crashdump_dir=os.path.abspath(f'{args.work_dir}/crashdumps'))

preproc = create_fsl_fmriprep_preproc(whichvol='first')

modelfit = create_modelfit_workflow()

modelspec = Node(model.SpecifyModel(), name="modelspec")

level1_workflow.connect([
    (preproc, modelspec, [('outputspec.highpassed_files', 'functional_runs')]),
    (modelspec, modelfit, [('session_info', 'inputspec.session_info')]),
    (preproc, modelfit, [('outputspec.highpassed_files',
                          'inputspec.functional_data')])
])

"""
Experiment specific components
------------------------------

"""

# Specify the subject directories
subject_list = [args.subject_id]
run_list = [args.run]

infosource = Node(
    IdentityInterface(fields=['subject_id', 'run']), name="infosource")
infosource.iterables = [('subject_id', subject_list),
                        ('run', run_list)]

# String template with {}-based strings
templates = {
    'func': 'derived/fmriprep/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-{run}_space-{space}_desc-preproc_bold.nii.gz',
    'mask': 'derived/fmriprep/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-{run}_space-{space}_desc-brain_mask.nii.gz',
    'confounds': 'derived/fmriprep/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-{run}_desc-confounds_regressors.tsv',
    'events': 'data/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-{run}_events.tsv',
}
# Create SelectFiles node
sf = Node(SelectFiles(templates),
          name='selectfiles')
# Location of the dataset folder
sf.inputs.base_directory = os.path.abspath(args.project_dir)
# Feed {}-based placeholder strings with values
sf.inputs.space = 'MNI152NLin2009cAsym'

"""
Setup a function that returns subject-specific information about the
experimental paradigm.
"""


def get_events(event_file, non_dummy_time, event_id):
    from nipype.interfaces.base import Bunch
    import pandas as pd
    events = pd.read_csv(event_file, sep='\t')

    # Filter by valid start times
    orig_len = len(events)
    events = events[events.onset >= non_dummy_time]

    # Keep our event_id consistant
    n_pruned = orig_len - len(events)
    event_id -= n_pruned
    # Fail at next step for pruned events, not here.
    if event_id < 0:
        return

    conditions = []
    onsets = []
    durations = []

    conditions.append('event_target')
    onsets.append([events.iloc[event_id]['onset']])
    durations.append([events.iloc[event_id]['duration']])

    conditions.append('event_nuisance')
    onsets.append(events.iloc[events.index != event_id]['onset'].values)
    durations.append(events.iloc[events.index != event_id]['duration'].values)

    output = Bunch(
        conditions=conditions,
        onsets=onsets,
        durations=durations
    )

    return output


getevents = Node(Function(input_names=['event_file', 'non_dummy_time', 'event_id'],
                          output_names=['events'],
                          function=get_events),
                 name='getevents')

event_list = range(60)
eventsource = Node(
    IdentityInterface(fields=['event_id']), name="eventsource")
eventsource.iterables = ('event_id', event_list)

"""
Setup the contrast structure that needs to be evaluated. This is a list of
lists. The inner list specifies the contrasts and has the following format -
[Name,Stat,[list of condition names],[weights on those conditions]. The
condition names must match the `names` listed in the `subjectinfo` function
described above.
"""

cont1 = ['Target>Nuisance', 'T', ['event_target', 'event_nusiance'], [1, -1]]
contrasts = [cont1]

hpcutoff = 60.0
TR = 0.8

modelspec.inputs.input_units = 'secs'
modelspec.inputs.time_repetition = TR
modelspec.inputs.high_pass_filter_cutoff = hpcutoff


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

findNonDummyTime = Node(Function(input_names=['regressors_file', 'tr'],
                                 output_names=['non_dummy_time'],
                                 function=FindNonDummyTime),
                        name='findNonDummyTime')
findNonDummyTime.inputs.tr = TR

extractconfounds = Node(ExtractConfounds(), name="extractconfounds")
# motion_6 or motion_24
extractconfounds.inputs.motion_params = 'motion_24'

level1_workflow.connect([
    (infosource, sf, [('subject_id', 'subject_id'),
                      ('run', 'run')]),
    # Event parsing
    (sf, findNonDummyTime, [('confounds', 'regressors_file')]),
    (findNonDummyTime, getevents, [('non_dummy_time', 'non_dummy_time')]),
    (sf, getevents, [('events', 'event_file')]),
    (eventsource, getevents, [('event_id', 'event_id')]),
    (getevents, modelspec, [('events', 'subject_info')]),

    (sf, preproc, [('func', 'inputspec.func'),
                   ('mask', 'inputspec.mask')]),
    (sf, extractconfounds, [('confounds', 'in_file')]),
    (extractconfounds, modelspec, [('out_file', 'realignment_parameters')]),
])


"""
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

substitutions = [('_mcf', ''),
                 ('_st', ''),
                 ('_flirt', ''),
                 ('.nii_mean_reg', '_mean'),
                 ('_modelestimate0/', ''),
                 ('.nii.par', '.par'),
                 ]
for run in range(1, 9):
    substitutions += [('_run_%d_subject_id_%s/' % (run, s), '') for s in subject_list]
for i in range(len(contrasts)):
    substitutions += [(f'_ztop{i}/', '')]
for f in fwhm:
    substitutions += [(f'_fwhm_{f}/', f'fwhm-{f}_')]
for event in event_list:
    substitutions += [(f'_event_id_{event}/', f'event-{event}_')]
sink.inputs.substitutions = substitutions


def trim_parameters(files):
    """
    For each of our 60 GLMs, we only want to keep the first parameter estimate.

    Discard the other 60 to save space.
    """
    if isinstance(files, list):
        return files[0][0]
    else:
        return files


# Connect DataSink with the relevant nodes
level1_workflow.connect([
    (modelfit, sink, [
        ('outputspec.copes', 'copes'),
        ('outputspec.varcopes', 'varcopes'),
        ('outputspec.dof_file', 'dof_file'),
        ('outputspec.pfiles', 'pfiles'),
        ('outputspec.zfiles', 'zfiles'),
        (('outputspec.parameter_estimates', trim_parameters), 'parameter_estimates')]),
])

"""
Execute the pipeline
--------------------

The code discussed above sets up all the necessary data structures with
appropriate parameters and the connectivity between the processes, but does not
generate any output. To actually run the analysis on the data the
``nipype.pipeline.engine.Pipeline.Run`` function needs to be called.
"""

level1_workflow.write_graph()
if not args.dry_run:
    if args.sge:
        level1_workflow.run(plugin='SGEGraph',
                            plugin_args={
                                'dont_resubmit_completed_jobs': True,
                                'qsub_args': '-l h_vmem=10G,s_vmem=9.5G -j y',
                            })
    elif args.n_procs:
        level1_workflow.run(plugin='MultiProc',
                            plugin_args={
                                'n_procs': args.n_procs
                            })
    else:
        level1_workflow.run()
