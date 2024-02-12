#!/bin/bash
set -euo pipefail

# Get PROJECT_DIR
source ../../config.cfg

DATE=$(date +%Y-%m-%d)
FMRIPREP_DIR=$PROJECT_DIR/derived/fmriprep
OUTPUT_DIR=$PROJECT_DIR/derived/rsa_lower_bound_pymvpa_random_null-${DATE}
mkdir -p $OUTPUT_DIR

# Cluster Setup
# Number of cores to request
N_CORES=1
# Amount of memory to request
MEMORY_GB=80.0
# Output script name
SCRIPT_BASE_NAME=rsa_lower_bound_pymvpa_random_null.sh

# Hard and soft memory limits
H_VMEM=$(echo "$MEMORY_GB * 1.0" | bc)
S_VMEM=$(echo "$MEMORY_GB * 1.0 - 0.5" | bc)

# Job script
ALL_JOB_SCRIPT="run_all.$SCRIPT_BASE_NAME"
cat << EOF > $ALL_JOB_SCRIPT
#!/bin/bash
set -euo pipefail
EOF

mkdir -p scripts

comparison_metric=pearson
pairwise_metric=euclidean
n_permutations=1000
for seed in $(seq 1 $n_permutations); do
    # Create qsub script
    SCRIPT_NAME="scripts/${pairwise_metric}.${comparison_metric}.${seed}.$SCRIPT_BASE_NAME"
    cat << EOF > $SCRIPT_NAME
#!/bin/bash
#$ -j y
#$ -l h_vmem=${H_VMEM}G,s_vmem=${S_VMEM}G
#$ -o ${PROJECT_DIR}/logs/\$JOB_NAME.\$JOB_ID
# -m ea
# -M $EMAIL
#$ -pe threaded $N_CORES

set -euo pipefail

python $PROJECT_DIR/scripts/rsa/rsa_lower_bound_pymvpa_random_null.py $PROJECT_DIR $OUTPUT_DIR $pairwise_metric $comparison_metric $seed

EOF
    chmod u+x $SCRIPT_NAME
    echo "qsub $PWD/$SCRIPT_NAME" >> $ALL_JOB_SCRIPT
done
chmod u+x $ALL_JOB_SCRIPT

# Script will have to be manually run
cp rsa_lower_bound_pymvpa_random_null_stats.sh $OUTPUT_DIR/
