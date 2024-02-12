#!/bin/bash
set -euo pipefail

####
# Create a report of max and mean framewise displacement for each subject.
# Requires that fMRIPrep has already been run
####

# Project Setup
# Get PROJECT_DIR
source ../../config.cfg
FMRIPREP_DIR=$PROJECT_DIR/derived/fmriprep

printf "subject\tmax_fd\tmean_fd\n" > fd_report.tsv
for report in $FMRIPREP_DIR/*.html; do
	subject="${report:(-11):6}"
	max_fd=$(grep mean "$FMRIPREP_DIR/sub-${subject}/ses-1/figures/sub-${subject}_ses-1_task-rest_acq-pre_run-1_desc-carpetplot_bold.svg" | tail -n 1 | awk '{print $3}')
	mean_fd=$(grep mean "$FMRIPREP_DIR/sub-${subject}/ses-1/figures/sub-${subject}_ses-1_task-rest_acq-pre_run-1_desc-carpetplot_bold.svg" | tail -n 1 | awk '{print $6}')
	printf "$subject\t$max_fd\t$mean_fd\n" >> fd_report.tsv
done
