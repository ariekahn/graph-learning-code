#!/bin/bash
#set -euo pipefail

# Get PROJECT_DIR
source ../../config.cfg

DATA_DIR=$PROJECT_DIR/data

cd $DATA_DIR
for sub in `find $DATA_DIR -maxdepth 1 -mindepth 1 -type d -name "sub-*" -printf "%f\n"`
do
    ls $sub/ses-1 > /dev/null
    if [[ -d $sub/ses-1 ]]; then
        ls $sub/ses-1/anat/${sub}_ses-1_T1w.nii.gz > /dev/null
        ls $sub/ses-1/func/${sub}_ses-1_task-graphlearning_run-1_bold.nii.gz > /dev/null
        ls $sub/ses-1/func/${sub}_ses-1_task-graphlearning_run-2_bold.nii.gz > /dev/null
        ls $sub/ses-1/func/${sub}_ses-1_task-graphlearning_run-3_bold.nii.gz > /dev/null
        ls $sub/ses-1/func/${sub}_ses-1_task-graphlearning_run-4_bold.nii.gz > /dev/null
        ls $sub/ses-1/func/${sub}_ses-1_task-graphlearning_run-5_bold.nii.gz > /dev/null
        ls $sub/ses-1/func/${sub}_ses-1_task-rest_run-1_bold.nii.gz > /dev/null
        ls $sub/ses-1/func/${sub}_ses-1_task-rest_run-2_bold.nii.gz > /dev/null
        ls $sub/ses-1/fmap/${sub}_ses-1_acq-func_dir-PA_run-1_epi.nii.gz > /dev/null
        ls $sub/ses-1/fmap/${sub}_ses-1_acq-func_dir-AP_run-1_epi.nii.gz > /dev/null
    fi
    ls $sub/ses-2 > /dev/null
    if [[ -d $sub/ses-2 ]]; then
        ls $sub/ses-2/anat/${sub}_ses-2_T2w.nii.gz > /dev/null
        ls $sub/ses-2/func/${sub}_ses-2_task-graphrepresentation_run-1_bold.nii.gz > /dev/null
        ls $sub/ses-2/func/${sub}_ses-2_task-graphrepresentation_run-2_bold.nii.gz > /dev/null
        ls $sub/ses-2/func/${sub}_ses-2_task-graphrepresentation_run-3_bold.nii.gz > /dev/null
        ls $sub/ses-2/func/${sub}_ses-2_task-graphrepresentation_run-4_bold.nii.gz > /dev/null
        ls $sub/ses-2/func/${sub}_ses-2_task-graphrepresentation_run-5_bold.nii.gz > /dev/null
        ls $sub/ses-2/func/${sub}_ses-2_task-graphrepresentation_run-6_bold.nii.gz > /dev/null
        ls $sub/ses-2/func/${sub}_ses-2_task-graphrepresentation_run-7_bold.nii.gz > /dev/null
        ls $sub/ses-2/func/${sub}_ses-2_task-graphrepresentation_run-8_bold.nii.gz > /dev/null
        ls $sub/ses-2/func/${sub}_ses-2_task-graphlocalizer_bold.nii.gz > /dev/null
        ls $sub/ses-2/fmap/${sub}_ses-2_acq-func_dir-PA_run-1_epi.nii.gz > /dev/null
        ls $sub/ses-2/fmap/${sub}_ses-2_acq-func_dir-AP_run-1_epi.nii.gz > /dev/null
    fi
done
