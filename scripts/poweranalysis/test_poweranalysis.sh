#!/bin/bash
#$ -j y
#$ -l h_vmem=20.0G
#$ -o /cbica/projects/GraphLearning/project/logs/$JOB_NAME.$JOB_ID.$SGE_TASK_ID
# -m ea
#$ -pe threaded 10
#$ -t 1-30

set -euo pipefail

project_dir=/cbica/projects/GraphLearning/project
r_img="$project_dir/singularity/R-base-4.0.2.simg"
poweranalysis_script=$project_dir/scripts/poweranalysis/poweranalysis.R
output_dir=$project_dir/derived/poweranalysis

mkdir -p $output_dir

export SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs:$HOME/singularity_local/lib
export SINGULARITYENV_PKG_CONFIG_PATH=$HOME/singularity_local/lib/pkgconfig
export SINGULARITYENV_OMP_THREAD_LIMIT=1
export SINGULARITYENV_OMP_NUM_THREADS=1
singularity run --cleanenv $r_img Rscript --vanilla $poweranalysis_script $output_dir 32 2
