#!/bin/bash
set -euo pipefail

# Bypass the FEAT registration, since we've already registered the data
# in fMRIprep. In the level 1 analysis, we tell it not to perform
# registration, but now for the level 2 analysis, we want it to think
# we /did/ register the data, by copying over an identity transform.
# See: https://mumfordbrainstats.tumblr.com/post/166054797696/feat-registration-workaround

# Get PROJECT_DIR
source ../../config.cfg

subjects=( $( cut -d$'\t' -f1 $PROJECT_DIR/extra/behavioral_list.tsv ) )

for report in $PROJECT_DIR/derived/fmriprep/*.html; do
    subject=${report:(-11):6}

    # Make sure we have behavioral info for them as well
    if printf '%s\n' ${subjects[@]} | grep -q -P "^$subject$"; then
        echo $subject

        for run in {1..8}; do
            FEATDIR="$PROJECT_DIR/derived/fsl/sub-$subject/run${run}.feat"
 
            cd $FEATDIR
            # Check if we already created a backup directory.
            if [[ ! -d backup ]]; then
                # If reg_standard already exists from a level 2 analysis, remove it
                rm -rf reg_standard
                # Back up our old registration data
                mkdir backup
                mv reg/*.mat backup/
                mv reg/standard.nii.gz backup/
                # Copy over an identity matrix
                cp "$FSLDIR/etc/flirtsch/ident.mat" reg/example_func2standard.mat
                # Use the functional mean as our standard template
                cp mean_func.nii.gz reg/standard.nii.gz
            fi
        done
    fi
done
