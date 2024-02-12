# MVPA analysis

## Installation
In order for nipype to run using the SGEGraph plugin, it needs to be able to import
custom classes, which requires that we install these scripts as a python library.

To do so, using the same python environment that will execute on SGE,
run `pip install -e .`. The `-e` flag installs it in "editable" mode,
which means that any changes you make to the scripts in this directory
will instantly affect any code you run, rather than needing to reinstall.

## Running

`make_mvpa.sh` creates a series of shell scripts that run `fsl_representation_mvpa.py`
with arguments __SUBJECT__ and __RUN__.

`fsl_representation_mvpa`: This takes two arguments,
`subject_id` (e.g. 'GLS003') and `run` (e.g. 1-8)

This runs a first-level FSL workflow

Two parts:

1. preproc
2. modelfit

**Note on LS-S errors**:
`feat_representation_lss` may produce errors for early events if volumes have been trimmed,
as is the case with non-steady-state volumes. The approach models each event
as a standalone regressor versus all other combined events, which will fail if
that event isn't contained in the data. The script here tries and fails to model such events.
So long as no subsequent events produce errors, this can be safely ignored.

__preproc__ runs smoothing, masking, and highpass filtering
of the original BOLD series.

Each BOLD run is comprised of 60 shape events. We want to extract the
corresponding response to each event, using a GLM.

I think what actually happens is each point is defined by the beta weights,
as in the raw parameter estimates, rather than on any contrast of those weights
given the goal isn’t to threshold voxels (you typically select a n ROI through
another method), but essentially just get a response profile of each voxel to
each stimulus presentation

We're using an LS-S approach, where each event is fit with a separate GLM,
with one regressor for that event, and one other regressor that covers
all other events.



After preprocessing, we want to run 

__modelfit__ 

<img src="fsl/workingdir/representation_mvpa/graph.png" width="400px">


