#!/bin/bash
set -euo pipefail

# Get PROJECT_DIR
source ../../config.cfg

DATE=$(date +%Y-%m-%d)
FMRIPREP_DIR=$PROJECT_DIR/derived/fmriprep
CODE_DIR=$PROJECT_DIR/scripts/rsa
OUTPUT_DIR=$PROJECT_DIR/derived/cosmo_make_rdm_searchlight_per_run-${DATE}
mkdir -p $OUTPUT_DIR

# Cluster Setup
# Number of cores to request
N_CORES=1
# Amount of memory to request
MEMORY_GB=10.0
# Output script name
SCRIPT_BASE_NAME=cosmo_make_rdm_searchlight_per_run.sh

# Hard and soft memory limits
H_VMEM=`echo "$MEMORY_GB * 1.0" | bc`
S_VMEM=`echo "$MEMORY_GB * 1.0 - 0.5" | bc`

# Job script
ALL_JOB_SCRIPT="run_all.$SCRIPT_BASE_NAME"
cat << EOF > $ALL_JOB_SCRIPT
#!/bin/bash
set -euo pipefail
if [[ cosmo_make_rdm_searchlight.m -nt cosmo_make_rdm_searchlight_per_run ]]; then
  echo .m file is newer than mcc output
  echo run \\\`mcc -m cosmo_make_rdm_searchlight_per_run.m\\\`
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

LD_PRELOAD=/cbica/software/external/matlab/R2018A/sys/os/glnxa64/libstdc++.so.6.0.22 ./cosmo_make_rdm_searchlight_per_run $subject 1 $PROJECT_DIR $OUTPUT_DIR/sub-$subject $N_CORES
LD_PRELOAD=/cbica/software/external/matlab/R2018A/sys/os/glnxa64/libstdc++.so.6.0.22 ./cosmo_make_rdm_searchlight_per_run $subject 2 $PROJECT_DIR $OUTPUT_DIR/sub-$subject $N_CORES
LD_PRELOAD=/cbica/software/external/matlab/R2018A/sys/os/glnxa64/libstdc++.so.6.0.22 ./cosmo_make_rdm_searchlight_per_run $subject 3 $PROJECT_DIR $OUTPUT_DIR/sub-$subject $N_CORES
LD_PRELOAD=/cbica/software/external/matlab/R2018A/sys/os/glnxa64/libstdc++.so.6.0.22 ./cosmo_make_rdm_searchlight_per_run $subject 4 $PROJECT_DIR $OUTPUT_DIR/sub-$subject $N_CORES
LD_PRELOAD=/cbica/software/external/matlab/R2018A/sys/os/glnxa64/libstdc++.so.6.0.22 ./cosmo_make_rdm_searchlight_per_run $subject 5 $PROJECT_DIR $OUTPUT_DIR/sub-$subject $N_CORES
LD_PRELOAD=/cbica/software/external/matlab/R2018A/sys/os/glnxa64/libstdc++.so.6.0.22 ./cosmo_make_rdm_searchlight_per_run $subject 6 $PROJECT_DIR $OUTPUT_DIR/sub-$subject $N_CORES
LD_PRELOAD=/cbica/software/external/matlab/R2018A/sys/os/glnxa64/libstdc++.so.6.0.22 ./cosmo_make_rdm_searchlight_per_run $subject 7 $PROJECT_DIR $OUTPUT_DIR/sub-$subject $N_CORES
LD_PRELOAD=/cbica/software/external/matlab/R2018A/sys/os/glnxa64/libstdc++.so.6.0.22 ./cosmo_make_rdm_searchlight_per_run $subject 8 $PROJECT_DIR $OUTPUT_DIR/sub-$subject $N_CORES

EOF
    chmod u+x $SCRIPT_NAME
    echo "qsub $PWD/$SCRIPT_NAME" >> $ALL_JOB_SCRIPT
done
chmod u+x $ALL_JOB_SCRIPT
