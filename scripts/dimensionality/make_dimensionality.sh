#!/bin/bash
set -euo pipefail

# Get PROJECT_DIR
source ../../config.cfg

DATE=$(date +%Y-%m-%d)
FMRIPREP_DIR=$PROJECT_DIR/derived/fmriprep
OUTPUT_DIR=$PROJECT_DIR/derived/dimensionality-${DATE}
mkdir -p $OUTPUT_DIR

# Cluster Setup
# Number of cores to request
N_CORES=8
# Amount of memory to request
MEMORY_GB=6.0
# Output script name
SCRIPT_BASE_NAME=dimensionality.sh

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
for report in $FMRIPREP_DIR/*.html; do
    subject="${report:(-11):6}"
    echo $subject
    SUBJ_DIR=$FMRIPREP_DIR/$subject
    mkdir -p "$OUTPUT_DIR/sub-$subject"
    for roi in "postcentral-lh" "postcentral-rh" "loc-localized"; do
        # Create qsub script
        SCRIPT_NAME="scripts/sub-$subject.${roi}.$SCRIPT_BASE_NAME"
        cat << EOF > $SCRIPT_NAME
#!/bin/bash
#$ -j y
#$ -l h_vmem=${H_VMEM}G,s_vmem=${S_VMEM}G
#$ -o ${PROJECT_DIR}/logs/\$JOB_NAME.\$JOB_ID
# -m ea
# -M $EMAIL
#$ -pe threaded $N_CORES

set -euo pipefail

python $PROJECT_DIR/scripts/dimensionality/dimensionality.py $subject $roi $PROJECT_DIR $OUTPUT_DIR/sub-$subject $N_CORES

EOF
        chmod u+x $SCRIPT_NAME
        echo "qsub $PWD/$SCRIPT_NAME" >> $ALL_JOB_SCRIPT
    done
done
chmod u+x $ALL_JOB_SCRIPT
