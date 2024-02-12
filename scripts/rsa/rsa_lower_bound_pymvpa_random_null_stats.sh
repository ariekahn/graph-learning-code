#!/bin/bash
set -euo pipefail
basename="rdm_lower-bound_pairwise-metric-euclidean_radius-3_comparison-metric-pearson_ordering-random_"
graph=all
fslmerge -t ${basename}seed-combined_${graph}.nii.gz ${basename}seed-{1..1000}_${graph}.nii.gz
fslmaths ${basename}seed-combined_${graph}.nii.gz -abs -Xmax -Ymax -Zmax ${basename}volmaxes_${graph}.nii.gz
graph=modular
fslmerge -t ${basename}seed-combined_${graph}.nii.gz ${basename}seed-{1..1000}_${graph}.nii.gz
fslmaths ${basename}seed-combined_${graph}.nii.gz -abs -Xmax -Ymax -Zmax ${basename}volmaxes_${graph}.nii.gz
graph=lattice
fslmerge -t ${basename}seed-combined_${graph}.nii.gz ${basename}seed-{1..1000}_${graph}.nii.gz
fslmaths ${basename}seed-combined_${graph}.nii.gz -abs -Xmax -Ymax -Zmax ${basename}volmaxes_${graph}.nii.gz
