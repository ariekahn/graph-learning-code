#!/bin/bash
set -euo pipefail

# Our learning scans include both the
# rapid learning section as well as the 
# shape learning checks at the end of each
# section. We want to process the entire
# block as a single scan for fMRIprep,
# but here, before xcpengine, we want to split
# the file into the learning-only section

# Get PROJECT_DIR
source ../../config.cfg

FMRIPREP_DIR=$PROJECT_DIR/derived/fmriprep
BIDS_DIR=$PROJECT_DIR/data

# Cluster Setup
# Number of cores to request
N_CORES=1
# Amount of memory to request
MEMORY_GB=10.0
# Output script name
SCRIPT_BASE_NAME=split.sh

# Hard and soft memory limits
H_VMEM=`echo "$MEMORY_GB * 1.0" | bc`
S_VMEM=`echo "$MEMORY_GB * 1.0 - 0.5" | bc`

rm -f run_all.split.sh

for report in $PROJECT_DIR/derived/fmriprep/*.html; do
    subject="${report:(-11):6}"
    echo $subject

    # Create qsub script
    SCRIPT_NAME="scripts/sub-$subject.$SCRIPT_BASE_NAME"
    cat << EOF > $SCRIPT_NAME
#!/bin/bash
#$ -j y
#$ -l h_vmem=${H_VMEM}G,s_vmem=${S_VMEM}G
#$ -o ${PROJECT_DIR}/logs/\$JOB_NAME.\$JOB_ID
# -m ea
# -M $EMAIL
#$ -pe threaded $N_CORES

set -euo pipefail

python $PROJECT_DIR/scripts/xcpengine/split_learning.py $BIDS_DIR $FMRIPREP_DIR $subject

EOF
    chmod u+x $SCRIPT_NAME
    echo "qsub $PWD/$SCRIPT_NAME" >> run_all.split.sh
done
chmod u+x run_all.split.sh
