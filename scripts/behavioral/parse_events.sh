#!/bin/bash
# Parse behavioral events into BIDS data directory
# set -euo pipefail

# Get PROJECT_DIR
source ../../config.cfg

SCRIPTS_DIR=$PROJECT_DIR/scripts/behavioral
DATA_DIR=$PROJECT_DIR/data
BEHAVIORAL_DIR=$PROJECT_DIR/behavioral
# FSL_BASE_OUTPUT_DIR=$PROJECT_DIR/derived/fsl
BEHAVIORAL_LIST=$PROJECT_DIR/extra/behavioral_list.tsv

# How many volumes are we excluding?
n_volumes_skip=0

for sub in `find $DATA_DIR -maxdepth 1 -mindepth 1 -type d -name "sub-*" -printf "%f\n"`
do
    echo "Subject: $sub"
    sub=${sub:4} #take off the sub- part
    python $SCRIPTS_DIR/parse_events.py $DATA_DIR $BEHAVIORAL_DIR $BEHAVIORAL_LIST $sub
    # Below scripts are for FSL (non-nipype) usage
    # if [[ -d $DATA_DIR/sub-$sub/ses-2 ]]; then
	#     OUTPUT_DIR=$FSL_BASE_OUTPUT_DIR/sub-$sub/events
	#     mkdir -p $OUTPUT_DIR
	#     python $SCRIPTS_DIR/bids_learning_to_feat.py $DATA_DIR $OUTPUT_DIR $sub $n_volumes_skip 0.8
	#     python $SCRIPTS_DIR/bids_localizer_to_feat.py $DATA_DIR $OUTPUT_DIR $sub $n_volumes_skip 0.8
	#     python $SCRIPTS_DIR/bids_representation_to_feat.py $DATA_DIR $OUTPUT_DIR $sub $n_volumes_skip 0.8
    # fi
done
