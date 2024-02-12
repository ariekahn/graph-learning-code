#!/bin/bash
set -euo pipefail

# Get PROJECT_DIR
source ../../config.cfg

DATE=$(date +%Y-%m-%d)
FMRIPREP_DIR=$PROJECT_DIR/derived/fmriprep
WORK_DIR=$PROJECT_DIR/work/feat_localizer-${DATE}
OUTPUT_DIR=$PROJECT_DIR/derived/feat_localizer-${DATE}
UNVERSIONED_DIR="$PROJECT_DIR/derived/feat_localizer/"
relink=$(prompt_confirm "Link output to $VERSION? y/n")
if [[ "$relink" ]]; then
    ln -f -s "$UNVERSIONED_DIR" "$OUTPUT_DIR"
fi

# Cluster Setup
# Number of cores to request
N_CORES=1
# Amount of memory to request
MEMORY_GB=10.0
# Output script name
SCRIPT_BASE_NAME=feat_localizer.sh

# Hard and soft memory limits
H_VMEM=$(echo "$MEMORY_GB * 1.0" | bc)
S_VMEM=$(echo "$MEMORY_GB * 1.0 - 0.5" | bc)

# Job script
ALL_JOB_SCRIPT="run_all.$SCRIPT_BASE_NAME"
cat << EOF > $ALL_JOB_SCRIPT
#!/bin/bash
set -euo pipefail
EOF

for report in $FMRIPREP_DIR/*.html; do
    subject="${report:(-11):6}"
    echo $subject

    # Create qsub script
    SCRIPT_NAME="scripts/sub-$subject.$SCRIPT_BASE_NAME"
    SUBJ_DIR=$FMRIPREP_DIR/$subject
    cat << EOF > $SCRIPT_NAME
#!/bin/bash
#$ -j y
#$ -l h_vmem=${H_VMEM}G,s_vmem=${S_VMEM}G
#$ -o ${PROJECT_DIR}/logs/\$JOB_NAME.\$JOB_ID
# -m ea
# -M $EMAIL
#$ -pe threaded $N_CORES

set -euo pipefail

python $PROJECT_DIR/scripts/mvpa/feat_localizer.py $subject $PROJECT_DIR $WORK_DIR/sub-${subject} $OUTPUT_DIR/sub-${subject}

EOF
    chmod u+x $SCRIPT_NAME
    echo "qsub $PWD/$SCRIPT_NAME" >> $ALL_JOB_SCRIPT
done
chmod u+x $ALL_JOB_SCRIPT
