#!/bin/bash
#$ -j y
#$ -l h_vmem=32.0G,s_vmem=31.0G
#$ -o /cbica/projects/GraphLearning/project/logs/$JOB_NAME.$JOB_ID
set -euo pipefail

# Get PROJECT_DIR
source ../../config.cfg

IMAGE_DIR=$PROJECT_DIR/singularity
VERSION="0.15.2"

mkdir -p $IMAGE_DIR
singularity build $IMAGE_DIR/mriqc-$VERSION.simg docker://poldracklab/mriqc:$VERSION
