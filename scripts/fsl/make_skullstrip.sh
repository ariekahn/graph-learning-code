#!/bin/bash
set -euo pipefail

# fMRIprep already generated brainmasks for us,
# so we can use those to strip the data and
# skip a step in FSL. We apply the brainmask
# to the MNI152NLin6Asym_res-2 functional data, and use
# that as our FEAT input.

# Get PROJECT_DIR
source ../../config.cfg

FMRIPREP_DIR=$PROJECT_DIR/derived/fmriprep
BIDS_DIR=$PROJECT_DIR/data

# Singularity Setup
# Path to singularity images
IMAGE_DIR=$PROJECT_DIR/singularity
# Name of image to use
IMAGE_NAME=fsl-6.0.3.simg

# Image space
SPACE=MNI152NLin6Asym_res-2

# Cluster Setup
# Number of cores to request
N_CORES=1
# Amount of memory to request
MEMORY_GB=10.0
# Output script name
SCRIPT_BASE_NAME=skullstrip.sh
# Run command
CMD="singularity"
# CMD="memrec -o memprofile.\${JOB_ID}.\${SGE_TASK_ID} $CMD"

# Hard and soft memory limits
H_VMEM=`echo "$MEMORY_GB * 1.0" | bc`
S_VMEM=`echo "$MEMORY_GB * 1.0 - 0.5" | bc`

rm -f run_all.skullstrip.sh

for report in $PROJECT_DIR/derived/fmriprep/*.html; do
    subject="sub-${report:(-11):6}"
    echo $subject

    # Create qsub script
    SCRIPT_NAME="scripts/$subject.$SCRIPT_BASE_NAME"
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

# learning
for run in {1..5}; do
	orig=${SUBJ_DIR}/ses-1/func/${subject}_ses-1_task-graphlearning_run-\${run}_space-${SPACE}_desc-preproc_bold.nii.gz
	mask=${SUBJ_DIR}/ses-1/func/${subject}_ses-1_task-graphlearning_run-\${run}_space-${SPACE}_desc-brain_mask.nii.gz
	output=${SUBJ_DIR}/ses-1/func/${subject}_ses-1_task-graphlearning_run-\${run}_space-${SPACE}_desc-preproc_bold_masked.nii.gz
	singularity exec --cleanenv $IMAGE_DIR/$IMAGE_NAME fslmaths \$orig -mas \$mask \$output
done

representation
for run in {1..8}; do
	orig=${SUBJ_DIR}/ses-2/func/${subject}_ses-2_task-graphrepresentation_run-\${run}_space-${SPACE}_desc-preproc_bold.nii.gz
	mask=${SUBJ_DIR}/ses-2/func/${subject}_ses-2_task-graphrepresentation_run-\${run}_space-${SPACE}_desc-brain_mask.nii.gz
	output=${SUBJ_DIR}/ses-2/func/${subject}_ses-2_task-graphrepresentation_run-\${run}_space-${SPACE}_desc-preproc_bold_masked.nii.gz
	singularity exec --cleanenv $IMAGE_DIR/$IMAGE_NAME fslmaths \$orig -mas \$mask \$output
done

# localizer
orig=${SUBJ_DIR}/ses-2/func/${subject}_ses-2_task-graphlocalizer_space-${SPACE}_desc-preproc_bold.nii.gz
mask=${SUBJ_DIR}/ses-2/func/${subject}_ses-2_task-graphlocalizer_space-${SPACE}_desc-brain_mask.nii.gz
output=${SUBJ_DIR}/ses-2/func/${subject}_ses-2_task-graphlocalizer_space-${SPACE}_desc-preproc_bold_masked.nii.gz
singularity exec --cleanenv $IMAGE_DIR/$IMAGE_NAME fslmaths \$orig -mas \$mask \$output

EOF
    chmod u+x $SCRIPT_NAME
    echo "qsub $PWD/$SCRIPT_NAME" >> run_all.skullstrip.sh
done
chmod u+x run_all.skullstrip.sh
