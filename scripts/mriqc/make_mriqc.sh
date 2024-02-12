#!/bin/bash
set -euo pipefail

# Project Setup
# Get PROJECT_DIR
source ../../config.cfg

# Singularity Setup
# Path to singularity images
IMAGE_DIR="$PROJECT_DIR/singularity"
# Name of image to use
VERSION="0.15.2"
IMAGE_NAME="mriqc-$VERSION.simg"
OUTPUT_DIR="$PROJECT_DIR/derived/mriqc-$VERSION/"
mkdir -p "$OUTPUT_DIR"

# Cluster Setup
# Number of cores to request
N_CORES=1
# Amount of memory to request
MEMORY_GB=40
# Output script name
SCRIPT_BASE_NAME="mriqc-$VERSION.sh"
# Run command
CMD="singularity"

# Hard and soft memory limits
H_VMEM=$(echo "$MEMORY_GB * 1.0" | bc)
S_VMEM=$(echo "$MEMORY_GB * 1.0 - 0.5" | bc)

# Subject list (csv)
SUBJECT_LIST="$PROJECT_DIR/extra/scan_list.tsv"
# participant names
SUBJECTS=( $(cut -f1 "$SUBJECT_LIST" | tail -n +2) )

###########################################
# Create header for the run_all script
# The record-qsub function keeps a list of all jobids
###########################################
RUN_ALL="run_all.$SCRIPT_BASE_NAME"
cat << EOF > "$RUN_ALL"
#!/bin/bash
set -euo pipefail
joblist=""
function record-qsub {
    jobid=\$(qsub \$1 | awk '{print \$3'})
    joblist="\$joblist,\$jobid"
}
EOF


###########################################
# Create a script for each subject
###########################################
mkdir -p scripts
for subject in "${SUBJECTS[@]}"; do
    echo "$subject"
    SCRIPT_NAME="scripts/$subject.$SCRIPT_BASE_NAME"
    WORK_DIR="$PROJECT_DIR/work/mriqc-${VERSION}/$subject"
    mkdir -p "$WORK_DIR"

    cat << EOF > "$SCRIPT_NAME"
#!/bin/bash
#$ -j y
#$ -l h_vmem=${H_VMEM}G,s_vmem=${S_VMEM}G
#$ -o ${PROJECT_DIR}/logs/\$JOB_NAME.\$JOB_ID.\$TASK_ID
#$ -m ea
#$ -M $EMAIL
#$ -pe threaded $N_CORES

set -euo pipefail

SIMG="${IMAGE_DIR}/${IMAGE_NAME}"

$CMD run -B \$SBIA_TMPDIR:/tmp --cleanenv \$SIMG \\
    "$PROJECT_DIR/data/" \\
    "$OUTPUT_DIR" \\
    participant \\
    --participant-label "$subject" \\
    --mem_gb $MEMORY_GB \\
    -w "$WORK_DIR"
EOF
    chmod u+x "$SCRIPT_NAME"
    echo "record-qsub \"$PWD/$SCRIPT_NAME\"" >> $RUN_ALL
done


###########################################
# Create a group script to run at the end
###########################################
SCRIPT_NAME="scripts/group.$SCRIPT_BASE_NAME"
WORK_DIR="$PROJECT_DIR/work/mriqc-${VERSION}/group"
mkdir -p "$WORK_DIR"
cat << EOF > $SCRIPT_NAME
#!/bin/bash
#$ -j y
#$ -l h_vmem=${H_VMEM}G,s_vmem=${S_VMEM}G
#$ -o ${PROJECT_DIR}/logs/\$JOB_NAME.\$JOB_ID.\$TASK_ID
#$ -m ea
#$ -M $EMAIL
#$ -pe threaded $N_CORES

set -euo pipefail

SIMG="${IMAGE_DIR}/${IMAGE_NAME}"

$CMD run -B \$SBIA_TMPDIR:/tmp --cleanenv \$SIMG \\
    "$PROJECT_DIR/data/" \\
    "$OUTPUT_DIR" \\
    group \\
    --mem_gb $MEMORY_GB \\
    -w "$WORK_DIR"
EOF
chmod u+x "$SCRIPT_NAME"

echo "qsub -hold_jid \${joblist:1} \"$PWD/$SCRIPT_NAME\"" >> $RUN_ALL
chmod u+x "$RUN_ALL"
