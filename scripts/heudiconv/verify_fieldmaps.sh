#!/bin/bash
#set -euo pipefail

# Get PROJECT_DIR
source ../../config.cfg

DATA_DIR=$PROJECT_DIR/data

cd $DATA_DIR
for sub in `find $DATA_DIR -maxdepth 1 -mindepth 1 -type d -name "sub-*" -printf "%f\n"`
do
    if [[ -d $sub/ses-1 ]]; then
        grep -L IntendedFor $sub/ses-1/fmap/*.json
    fi
    if [[ -d $sub/ses-2 ]]; then
        grep -L IntendedFor $sub/ses-2/fmap/*.json
    fi
done
