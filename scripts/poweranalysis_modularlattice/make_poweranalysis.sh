#!/bin/bash
set -euo pipefail

# Get PROJECT_DIR
source ../../config.cfg

# Cluster Setup
# Number of cores to request
N_CORES=20
# Amount of memory to request
MEMORY_GB=50.0
# Output script name
SCRIPT_BASE_NAME=poweranalysis.sh

# Hard and soft memory limits
H_VMEM=$(echo "$MEMORY_GB * 1.0" | bc)
S_VMEM=$(echo "$MEMORY_GB * 1.0 - 0.5" | bc)

# Job script
ALL_JOB_SCRIPT="run_all.$SCRIPT_BASE_NAME"
cat << EOF > $ALL_JOB_SCRIPT
#!/bin/bash
set -euo pipefail
EOF

r_img="$PROJECT_DIR/singularity/R-tidyverse-4.0.4.simg"
nhb_data_dir="$PROJECT_DIR/derived/nhb_data/processed"
poweranalysis_script="$PROJECT_DIR/scripts/poweranalysis_modularlattice/poweranalysis.R"
install_deps_script="$PROJECT_DIR/scripts/poweranalysis_modularlattice/install_deps.R"
output_dir="$PROJECT_DIR/derived/poweranalysis_modularlattice"
local_R_dir="$PROJECT_DIR/scripts/poweranalysis_modularlattice/R"

mkdir -p $local_R_dir
mkdir -p $output_dir

export SINGULARITYENV_R_LIBS=$local_R_dir
singularity run --cleanenv $r_img Rscript --vanilla $install_deps_script

for n_subjects in {2..30}; do
    SCRIPT_NAME="scripts/n${n_subjects}.$SCRIPT_BASE_NAME"
    cat << EOF > $SCRIPT_NAME
#!/bin/bash
#$ -j y
#$ -l h_vmem=${H_VMEM}G
#$ -o ${PROJECT_DIR}/logs/\$JOB_NAME.\$JOB_ID
# -m ea
# -M $EMAIL
#$ -pe threaded $N_CORES

set -euo pipefail

export SINGULARITYENV_OMP_THREAD_LIMIT=1
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_R_LIBS=$local_R_dir
singularity run --cleanenv $r_img Rscript --vanilla $poweranalysis_script $nhb_data_dir $output_dir $n_subjects $(($N_CORES - 1))
EOF
    chmod u+x $SCRIPT_NAME
    echo "qsub $PWD/$SCRIPT_NAME" >> $ALL_JOB_SCRIPT
done
chmod u+x $ALL_JOB_SCRIPT
