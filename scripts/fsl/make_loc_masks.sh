#!/bin/bash
set -euo pipefail

# Here we combine the LOC localizer
# with the level two FEAT outputs to
# create a set of masked voxels in the LOC
# for each stimulus in the graph.

# Get PROJECT_DIR
source ../../config.cfg

FMRIPREP_DIR=$PROJECT_DIR/derived/fmriprep
FEAT_DIR=$PROJECT_DIR/derived/fsl

# How many voxels to include in the mask?
n_voxels=400

for report in $PROJECT_DIR/derived/fmriprep/*.html; do
    subject="${report:(-11):6}"
    if [ -d "$FEAT_DIR/sub-$subject/level-2.gfeat" ]; then
        echo $subject
        python loc_masks.py $FMRIPREP_DIR $FEAT_DIR $subject $n_voxels --dilate 1
    fi
done
