#!/bin/bash
set -euo pipefail

# fMRIprep generates motion and nuisance confounds for us.
# We want to use these in FEAT, so that we can avoid trimming
# our data, and also to regress out motion that FEAT doesn't
# know about. Here we take the .tsv files from fMRIprep and
# create a whitespace-separated set of columns, one for each
# confound.

# Get PROJECT_DIR
source ../../config.cfg

FMRIPREP_DIR=$PROJECT_DIR/derived/fmriprep

# How many volumes are we excluding?
n_volumes_skip=0

# Cluster Setup
# Number of cores to request
N_CORES=1
# Amount of memory to request
MEMORY_GB=10.0
# Output script name
SCRIPT_BASE_NAME=confounds.sh

# Hard and soft memory limits
H_VMEM=`echo "$MEMORY_GB * 1.0" | bc`
S_VMEM=`echo "$MEMORY_GB * 1.0 - 0.5" | bc`

rm -f run_all.confounds.sh

for report in $PROJECT_DIR/derived/fmriprep/*.html; do
    subject="${report:(-11):6}"
    echo $subject

    OUTPUT_DIR=$PROJECT_DIR/derived/fsl/sub-$subject/confounds
    mkdir -p $OUTPUT_DIR

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

python $PROJECT_DIR/scripts/fsl/confounds_to_fsl.py $FMRIPREP_DIR $OUTPUT_DIR $subject $n_volumes_skip

EOF
    chmod u+x $SCRIPT_NAME
    echo "qsub $PWD/$SCRIPT_NAME" >> run_all.confounds.sh
done
chmod u+x run_all.confounds.sh
