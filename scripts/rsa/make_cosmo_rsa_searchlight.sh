#!/bin/bash
set -euo pipefail

# Get PROJECT_DIR
source ../../config.cfg

DATE=$(date +%Y-%m-%d)
FMRIPREP_DIR=$PROJECT_DIR/derived/fmriprep
CODE_DIR=$PROJECT_DIR/scripts/rsa
OUTPUT_DIR=$PROJECT_DIR/derived/cosmo_rsa_searchlight-${DATE}
mkdir -p $OUTPUT_DIR

# Cluster Setup
# Number of cores to request
N_CORES=8
# Amount of memory to request
MEMORY_GB=40.0
# Output script name
SCRIPT_BASE_NAME=cosmo_rsa_searchlight.sh

# Hard and soft memory limits
H_VMEM=`echo "$MEMORY_GB * 1.0" | bc`
S_VMEM=`echo "$MEMORY_GB * 1.0 - 0.5" | bc`

# Job script
ALL_JOB_SCRIPT="run_all.$SCRIPT_BASE_NAME"
cat << EOF > $ALL_JOB_SCRIPT
#!/bin/bash
set -euo pipefail
if [[ cosmo_rsa_searchlight.m -nt cosmo_rsa_searchlight ]]; then
  echo .m file is newer than mcc output
  echo run \\\`mcc -m cosmo_rsa_searchlight.m\\\`
  exit 1
fi
EOF

mkdir -p scripts
for report in $FMRIPREP_DIR/*.html; do
    subject="${report:(-11):6}"
    echo $subject
    mkdir -p "$OUTPUT_DIR/sub-$subject"
    mkdir -p "$OUTPUT_DIR/sub-$subject/results"
    mkdir -p "$OUTPUT_DIR/sub-$subject/images"
    mkdir -p "$OUTPUT_DIR/sub-$subject/masks"

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

cd $CODE_DIR

LD_PRELOAD=/cbica/software/external/matlab/R2018A/sys/os/glnxa64/libstdc++.so.6.0.22 ./cosmo_rsa_searchlight $subject $PROJECT_DIR $OUTPUT_DIR/sub-$subject $N_CORES
#matlab -nodisplay -nodesktop -r "subject = '${subject}'; project_path = '${PROJECT_DIR}'; output_path = '${OUTPUT_DIR}/sub-$subject'; cosmo_rsa_searchlight; quit"

EOF
    chmod u+x $SCRIPT_NAME
    echo "qsub $PWD/$SCRIPT_NAME" >> $ALL_JOB_SCRIPT
done
chmod u+x $ALL_JOB_SCRIPT
