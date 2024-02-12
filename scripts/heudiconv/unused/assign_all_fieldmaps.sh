#!/bin/bash
# Assign fieldmaps for all already processed subjects
set -euo pipefail

# Get PROJECT_DIR
source ../../config.cfg

SCRIPTS_DIR=$PROJECT_DIR/scripts/heudiconv
DATA_DIR=$PROJECT_DIR/data

for sub in `find $DATA_DIR -maxdepth 1 -mindepth 1 -type d -name "sub-*" -printf "%f\n"`
do
    echo "Subject: $sub"
    sub=${sub:4} #take off the sub- part
    for ses in `find $DATA_DIR/sub-$sub -maxdepth 1 -mindepth 1 -type d -name "ses-*" -printf "%f\n"`
    do
        echo "Session: $ses"
        ses=${ses:4} #take off the ses- part
        python ${SCRIPTS_DIR}/assign_fieldmaps.py $DATA_DIR $sub $ses --overwrite --database_path=$PROJECT_DIR/work/bidsdb
    done
done
