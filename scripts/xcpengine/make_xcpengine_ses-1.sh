#!/bin/bash
set -euo pipefail

# Project Setup
# Get PROJECT_DIR
source ../../config.cfg

# Singularity Setup
# Path to singularity images
IMAGE_DIR=$PROJECT_DIR/singularity
# Name of image to use
IMAGE_NAME=xcpengine-latest.simg

## XCP Config
## Design file
## design=fc-aroma
design=fc-acompcor
# design=fc-36p
## Functional space
# func_space=T1w
# func_space=MNI152NLin2009cAsym
func_space=MNI152NLin6Asym_res-2
# func_space=MNI152NLin6Asym

DATE=$(date +%Y-%m-%d)
FMRIPREP_DIR=$PROJECT_DIR/derived/fmriprep
WORK_DIR="$PROJECT_DIR/work/xcpEngine_${func_space}_${design}-${DATE}"
OUTPUT_DIR="$PROJECT_DIR/derived/xcpEngine_${func_space}_${design}-${DATE}"

# Cluster Setup
# Number of cores to request
N_CORES=2
# Amount of memory to request
MEMORY_GB=30.0
# Output script name
SCRIPT_BASE_NAME="xcp_parallel.${func_space}.${design}.sh"
# Run command
CMD="singularity"
# CMD="memrec -o memprofile.\${JOB_ID}.\${SGE_TASK_ID} $CMD"

# Hard and soft memory limits
H_VMEM=$(echo "$MEMORY_GB * 1.0" | bc)
S_VMEM=$(echo "$MEMORY_GB * 1.0 - 0.5" | bc)

RUN_ALL="run_all.$SCRIPT_BASE_NAME"
cat << EOF > $RUN_ALL
#!/bin/bash
set -euo pipefail
EOF

mkdir -p scripts

# Create a file of our scans to process
XCP_TEMP_DIR=$PROJECT_DIR/extra/temp
mkdir -p $XCP_TEMP_DIR
for report in $FMRIPREP_DIR/*.html; do
    subject=${report: -11:-5}
    subject="sub-$subject"
    echo $subject
    # Create temporary cohort file
    XCP_TEMP_FILE="${XCP_TEMP_DIR}/${subject}_${func_space}_${design}.csv"
    echo 'id0,id1,img' > $XCP_TEMP_FILE
    echo "${subject},rest-1,${subject}/ses-1/func/${subject}_ses-1_task-rest_run-1_space-${func_space}_desc-preproc_bold.nii.gz" >> $XCP_TEMP_FILE
    echo "${subject},rest-2,${subject}/ses-1/func/${subject}_ses-1_task-rest_run-2_space-${func_space}_desc-preproc_bold.nii.gz" >> $XCP_TEMP_FILE
    echo "${subject},rest-3,${subject}/ses-1/func/${subject}_ses-1_task-rest_run-3_space-${func_space}_desc-preproc_bold.nii.gz" >> $XCP_TEMP_FILE
    echo "${subject},rest-4,${subject}/ses-1/func/${subject}_ses-1_task-rest_run-4_space-${func_space}_desc-preproc_bold.nii.gz" >> $XCP_TEMP_FILE
    echo "${subject},task-1,${subject}/ses-1/func/${subject}_ses-1_task-graphlearning_run-1_space-${func_space}_desc-preproc_bold.nii.gz" >> $XCP_TEMP_FILE
    echo "${subject},task-2,${subject}/ses-1/func/${subject}_ses-1_task-graphlearning_run-2_space-${func_space}_desc-preproc_bold.nii.gz" >> $XCP_TEMP_FILE
    echo "${subject},task-3,${subject}/ses-1/func/${subject}_ses-1_task-graphlearning_run-3_space-${func_space}_desc-preproc_bold.nii.gz" >> $XCP_TEMP_FILE
    echo "${subject},task-4,${subject}/ses-1/func/${subject}_ses-1_task-graphlearning_run-4_space-${func_space}_desc-preproc_bold.nii.gz" >> $XCP_TEMP_FILE
    echo "${subject},task-5,${subject}/ses-1/func/${subject}_ses-1_task-graphlearning_run-5_space-${func_space}_desc-preproc_bold.nii.gz" >> $XCP_TEMP_FILE

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

set -o errexit
set -o pipefail
set -o nounset

SIMG="${IMAGE_DIR}/${IMAGE_NAME}"

$CMD run --cleanenv -B \$SBIA_TMPDIR:/tmp \$SIMG \\
  -c $XCP_TEMP_FILE \\
  -d $PROJECT_DIR/extra/xcp_designs/${design}.dsn \\
  -o $OUTPUT_DIR \\
  -r $FMRIPREP_DIR \\
  -i $WORK_DIR \\
  -t 2
EOF
    chmod u+x $SCRIPT_NAME
    echo "qsub $PWD/$SCRIPT_NAME" >> $RUN_ALL
done
chmod u+x $RUN_ALL
