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
from nipype import SelectFiles, Node, IdentityInterface, Workflow, DataSink, Function

from graphlearning_mvpa.preprocess import create_fsl_fmriprep_preproc
from graphlearning_mvpa.estimate import create_modelfit_workflow
from graphlearning_mvpa.util import ExtractConfounds, FindNonDummyTime

parser = argparse.ArgumentParser('Run FSL Representation analysis in LS-A style')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('run', type=int, help='Run to process')
parser.add_argument('project_dir', type=str, help='directory containing data and derived subfolders')
parser.add_argument('work_dir', type=str, help='Temporary work directory')
parser.add_argument('output_dir', type=str, help='Location for output files')
parser.add_argument('--dry_run', action='store_true', help="Attempt to create workflow but don't run.")
parser.add_argument('--debug', action='store_true', help="Run nipype in debug mode")
parser.add_argument('--sge', action='store_true', help="Run using SGE")
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

level1_workflow = Workflow(name='feat_representation_lsa')
level1_workflow.base_dir = os.path.abspath(args.work_dir)
level1_workflow.config['execution'] = dict(
    crashdump_dir=os.path.abspath(f'{args.work_dir}/crashdumps'))

preproc = create_fsl_fmriprep_preproc(whichvol='first')

modelfit = create_modelfit_workflow(keep_res4d=True)

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


def get_events_combined(event_file, non_dummy_time):
    from nipype.interfaces.base import Bunch
    import pandas as pd
    events = pd.read_csv(event_file, sep='\t')

    # Filter by valid start times
    events = events[events.onset >= non_dummy_time]

    conditions = []
    onsets = []
    durations = []

    for node in range(15):
        conditions.append(f'node_{node}')
        onsets.append(events[events['node'] == node]['onset'].values)
        durations.append(events[events['node'] == node]['duration'].values)

    output = Bunch(
        conditions=conditions,
        onsets=onsets,
        durations=durations
    )

    return output


getevents = Node(Function(input_names=['event_file', 'non_dummy_time'],
                          output_names=['events'],
                          function=get_events_combined),
                 name='getevents')

"""
Setup the contrast structure that needs to be evaluated. This is a list of
lists. The inner list specifies the contrasts and has the following format -
[Name,Stat,[list of condition names],[weights on those conditions]. The
condition names must match the `names` listed in the `subjectinfo` function
described above.
"""

cont_all = ['All Nodes', 'T', [f'node_{i}' for i in range(15)], [1] * 15]
cont1 = ['Node 1', 'T', [f'node_{i}' for i in range(15)], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
cont2 = ['Node 2', 'T', [f'node_{i}' for i in range(15)], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
cont3 = ['Node 3', 'T', [f'node_{i}' for i in range(15)], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
cont4 = ['Node 4', 'T', [f'node_{i}' for i in range(15)], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
cont5 = ['Node 5', 'T', [f'node_{i}' for i in range(15)], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
cont6 = ['Node 6', 'T', [f'node_{i}' for i in range(15)], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
cont7 = ['Node 7', 'T', [f'node_{i}' for i in range(15)], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
cont8 = ['Node 8', 'T', [f'node_{i}' for i in range(15)], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
cont9 = ['Node 9', 'T', [f'node_{i}' for i in range(15)], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]
cont10 = ['Node 10', 'T', [f'node_{i}' for i in range(15)], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
cont11 = ['Node 11', 'T', [f'node_{i}' for i in range(15)], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
cont12 = ['Node 12', 'T', [f'node_{i}' for i in range(15)], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
cont13 = ['Node 13', 'T', [f'node_{i}' for i in range(15)], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
cont14 = ['Node 14', 'T', [f'node_{i}' for i in range(15)], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
cont15 = ['Node 15', 'T', [f'node_{i}' for i in range(15)], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
contrasts = [cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9, cont10, cont11, cont12, cont13, cont14, cont15, cont_all]

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

substitutions = [('_event_id_', ''),
                 ('_mcf', ''),
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
sink.inputs.substitutions = substitutions

# Connect DataSink with the relevant nodes
level1_workflow.connect([
    (modelfit, sink, [
        ('outputspec.copes', 'copes'),
        ('outputspec.varcopes', 'varcopes'),
        ('outputspec.dof_file', 'dof_file'),
        ('outputspec.pfiles', 'pfiles'),
        ('outputspec.zfiles', 'zfiles'),
        ('outputspec.res4d', 'res4d'),
        ('outputspec.parameter_estimates', 'parameter_estimates')]),
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
    else:
        level1_workflow.run()
