#!/bin/bash
set -euo pipefail

# Get PROJECT_DIR
source ../../config.cfg

DATE=$(date +%Y-%m-%d)
FMRIPREP_DIR=$PROJECT_DIR/derived/fmriprep

# Cluster Setup
# Number of cores to request
N_CORES=1
# Amount of memory to request
MEMORY_GB=10.0
# Output script name
SCRIPT_BASE_NAME=register_neuropythy.sh

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

SINGULARITYENV_SUBJECTS_DIR=/subjects singularity exec --cleanenv -B "$PROJECT_DIR/derived/freesurfer":/subjects "$PROJECT_DIR/singularity/neuropythy-latest.simg" python -m neuropythy atlas --atlases benson14 --volume-export --verbose sub-${subject}
EOF
    chmod u+x $SCRIPT_NAME
    echo "qsub $PWD/$SCRIPT_NAME" >> $ALL_JOB_SCRIPT
done

# For fsaverage
subject=fsaverage
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

SINGULARITYENV_SUBJECTS_DIR=/subjects singularity exec --cleanenv -B "$PROJECT_DIR/derived/freesurfer":/subjects "$PROJECT_DIR/singularity/neuropythy-latest.simg" python -m neuropythy atlas --atlases benson14 --volume-export --verbose fsaverage
EOF
chmod u+x $SCRIPT_NAME
echo "qsub $PWD/$SCRIPT_NAME" >> $ALL_JOB_SCRIPT
chmod u+x $ALL_JOB_SCRIPT
