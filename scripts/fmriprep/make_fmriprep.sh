#!/bin/bash
set -euo pipefail

# Project Setup
# Get PROJECT_DIR
source ../../config.cfg

# Singularity Setup
# Path to singularity images
IMAGE_DIR=$PROJECT_DIR/singularity
# Name of image to use
VERSION="20.1.1"
IMAGE_NAME="fmriprep-${VERSION}.simg"
WORK_DIR="$PROJECT_DIR/work/fmriprep-$VERSION/"
OUTPUT_DIR="$PROJECT_DIR/derived/fmriprep-$VERSION/"
mkdir -p "$WORK_DIR"
mkdir -p "$OUTPUT_DIR"
UNVERSIONED_DIR="$PROJECT_DIR/derived/fmriprep/"
relink=$(prompt_confirm "Link output to $VERSION? y/n")
if [[ "$relink" ]]; then
    ln -f -s "$UNVERSIONED_DIR" "$OUTPUT_DIR"
fi

# Cluster Setup
# Number of cores to request
N_CORES=6
# Amount of memory to request
MEMORY_GB=64
# Output script name
SCRIPT_BASE_NAME="fmriprep-${VERSION}.sh"
# Run command
CMD="singularity"

# Hard and soft memory limits
H_VMEM=$(echo "$MEMORY_GB * 1.0" | bc)
S_VMEM=$(echo "$MEMORY_GB * 1.0 - 0.5" | bc)

# Subject list (csv)
SUBJECT_LIST="$PROJECT_DIR/extra/scan_list.tsv"
# participant names
SUBJECTS=( $(cut -f1 $SUBJECT_LIST | tail -n +2) )

RUN_ALL="run_all.$SCRIPT_BASE_NAME"
rm -f $RUN_ALL
cat << EOF > $RUN_ALL
#!/bin/bash
set -euo pipefail
EOF

mkdir -p scripts

for subject in "${SUBJECTS[@]}"; do
    echo $subject
    SCRIPT_NAME="scripts/$subject.$SCRIPT_BASE_NAME"

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
    --nprocs $((N_CORES + 1)) \\
    --mem_mb $((MEMORY_GB * 1024)) \\
    --fs-license-file "$PROJECT_DIR/extra/license.txt" \\
    --fs-subjects-dir "$PROJECT_DIR/derived/freesurfer" \\
    --participant-label "$subject" \\
    -w "$WORK_DIR" \\
    --skull-strip-fixed-seed \\
    --bold2t1w-dof 9 \\
    --dummy-scans 10 \\
    --output-spaces MNI152NLin2009cAsym:res-native MNI152NLin6Asym:res-2

EOF

    chmod u+x "$SCRIPT_NAME"
    echo "qsub \"$PWD/$SCRIPT_NAME\"" >> $RUN_ALL
done
chmod u+x "$RUN_ALL"
