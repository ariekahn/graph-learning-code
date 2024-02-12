#!/bin/bash
set -euo pipefail

# The level two FEAT analysis combines
# the separate runs for each subject into
# a single map for each stimulus for that
# subject.

# Get PROJECT_DIR
source ../../config.cfg

FMRIPREP_DIR=$PROJECT_DIR/derived/fmriprep
BIDS_DIR=$PROJECT_DIR/data

TEMPLATE="$PROJECT_DIR/extra/feat_level-2_template.fsf"

SUBJECTS=( $( cut -d$'\t' -f1 $PROJECT_DIR/extra/behavioral_list.tsv ) )

# Singularity Setup
# Path to singularity images
IMAGE_DIR=$PROJECT_DIR/singularity
# Name of image to use
IMAGE_NAME=fsl-6.0.3.simg

# Cluster Setup
# Number of cores to request
N_CORES=1
# Amount of memory to request
MEMORY_GB=30.0
# Output script name
SCRIPT_BASE_NAME=feat_level-2.sh
# Run command
CMD="singularity"
# CMD="memrec -o memprofile.\${JOB_ID}.\${SGE_TASK_ID} $CMD"

# Hard and soft memory limits
H_VMEM=`echo "$MEMORY_GB * 1.0" | bc`
S_VMEM=`echo "$MEMORY_GB * 1.0 - 0.5" | bc`

# Create a file of our scans to process
FSL_TEMP_DIR=$PROJECT_DIR/extra/temp
mkdir -p $FSL_TEMP_DIR

RUN_ALL="run_all.$SCRIPT_BASE_NAME"
rm -f $RUN_ALL

for report in $PROJECT_DIR/derived/fmriprep/*.html; do
    subject=${report:(-11):6}
    echo $subject
    if printf '%s\n' ${SUBJECTS[@]} | grep -q -P "^$subject$"; then
        subject=sub-$subject

        # Create temporary cohort file
        FSL_TEMP_FILE="$FSL_TEMP_DIR/${subject}_feat_level-2.fsf"
        LEVEL_1_DIR=$PROJECT_DIR/derived/fsl/$subject
        # Make sure we're bypassing the level 1 registration given fmriprep
        for run in {1..8}; do
            if [[ ! -d $LEVEL_1_DIR/run${run}.feat/backup ]]; then
                echo "Missing registration bypass"
                exit 1
            fi
        done
        OUTPUT_DIR=$PROJECT_DIR/derived/fsl/$subject/level-2
        sed "s:FSLDIR:$FSLDIR:g; s:LEVEL_1_DIR:$LEVEL_1_DIR:g; s:OUTPUT_DIR:$OUTPUT_DIR:g" $TEMPLATE > $FSL_TEMP_FILE

        # Create qsub script
        SCRIPT_NAME="scripts/$subject.$SCRIPT_BASE_NAME"
        cat << EOF > $SCRIPT_NAME
#!/bin/bash
#$ -j y
#$ -l h_vmem=${H_VMEM}G,s_vmem=${S_VMEM}G
#$ -o ${PROJECT_DIR}/logs/\$JOB_NAME.\$JOB_ID
# -m ea
# -M $EMAIL
#$ -pe threaded $N_CORES

set -euo pipefail

SIMG=${IMAGE_DIR}/${IMAGE_NAME}

singularity exec --cleanenv \$SIMG feat $FSL_TEMP_FILE

EOF
        chmod u+x $SCRIPT_NAME
        echo "qsub $PWD/$SCRIPT_NAME" >> $RUN_ALL
    fi
done
chmod u+x $RUN_ALL
