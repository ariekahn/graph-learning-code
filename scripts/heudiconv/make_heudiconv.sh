#!/bin/bash
set -euo pipefail

# Project Setup
# Get PROJECT_DIR
source ../../config.cfg

SCRIPTS_DIR=$PROJECT_DIR/scripts/heudiconv
DATA_DIR=$PROJECT_DIR/data
WORK_DIR=$PROJECT_DIR/work

# Subject list (tsv)
HEURISTIC_LIST=$PROJECT_DIR/extra/heuristic_list.tsv

# Singularity Setup
# Path to singularity images
IMAGE_DIR=$PROJECT_DIR/singularity
# Name of image to use
IMAGE_NAME=heudiconv-0.8.0.simg
SIMG=$IMAGE_DIR/$IMAGE_NAME

# Cluster Setup
# Number of cores to request
N_CORES=2
# Amount of memory to request
MEMORY_GB=10.0
# Output script name
SCRIPT_NAME=run_all.heudiconv.sh
# Run command
# CMD="singularity"

# A few variables we need to compute
# Number of cluster jobs to submit
NSUBJECTS=$(($(wc -l < $HEURISTIC_LIST) - 1))
NJOBS=$(($NSUBJECTS * 2))
if [[ ${NSUBJECTS} == 0  ]]; then
    echo 'you dont have enough lines in your tsv file'
    exit 0
fi

# Hard and soft memory limits
H_VMEM=$(echo "$MEMORY_GB * 1.0" | bc)
S_VMEM=$(echo "$MEMORY_GB * 1.0 - 0.5" | bc)

cat << EOF > $SCRIPT_NAME
#!/bin/bash
#$ -j y
#$ -l h_vmem=${H_VMEM}G,s_vmem=${S_VMEM}G
#$ -o ${PROJECT_DIR}/logs/\$JOB_NAME.\$JOB_ID.\$TASK_ID
#$ -t 1-${NJOBS}
##$ -m ea
##$ -M $EMAIL
#$ -pe threaded $N_CORES

set -euo pipefail

# Want to only launch this once. Populates the top-level files
if [[ \$SGE_TASK_ID -eq 1 ]]; then
    qsub -hold_jid \$JOB_ID $PROJECT_DIR/scripts/heudiconv/heudiconv_populate.sh
    qsub -hold_jid \$JOB_ID $PROJECT_DIR/scripts/heudiconv/heudiconv_fieldmaps.sh
fi

SUBJECTS=( \$(cut -f1 $HEURISTIC_LIST) )
HEURISTICS_1=( \$(cut -f2 $HEURISTIC_LIST) )
HEURISTICS_2=( \$(cut -f3 $HEURISTIC_LIST) )

IND=\$((\$SGE_TASK_ID - 1))
SESSION=\$((\$IND / $NSUBJECTS + 1))
SUBJECTID=\$((\$IND % $NSUBJECTS + 1))

subject=\${SUBJECTS[\$SUBJECTID]}
if [[ \$SESSION -eq 1 ]]; then
    heuristic=$PROJECT_DIR/extra/heuristics/\${HEURISTICS_1[\$SUBJECTID]}
elif [[ \$SESSION -eq 2 ]]; then
    heuristic=$PROJECT_DIR/extra/heuristics/\${HEURISTICS_2[\$SUBJECTID]}
fi

echo "Subject: \$subject"
echo "Heuristic: \$heuristic"

# For now, passing in files rather than a dicom template,
# as otherwise heudiconv sees each tarball as a separate session
# Otherwise we could try something like
# -d "\$PROJECT_DIR/dicom/sub-{subject}/ses-{session}/*.tar.gz"
# singularity run -B \$SBIA_TMPDIR:/tmp --cleanenv $SIMG \\
heudiconv \\
    --files $PROJECT_DIR/dicom/sub-\$subject/ses-\$SESSION/*.tar.gz \\
    -s \$subject \\
    -o $DATA_DIR \\
    -b notop \\
    -ss \$SESSION \\
    -f \$heuristic
EOF

chmod u+x $SCRIPT_NAME

cat << EOF > heudiconv_populate.sh
#!/bin/bash
#$ -j y
#$ -l h_vmem=10.0G,s_vmem=9.5G
#$ -o /cbica/projects/GraphLearning/project/logs/\$JOB_NAME.\$JOB_ID

set -euo pipefail

PROJECT_DIR=/cbica/projects/GraphLearning/project

heuristic=$PROJECT_DIR/extra/heuristics/heuristic-ses-1.py

heudiconv \\
    --files $DATA_DIR/ \\
    -f \$heuristic \\
    --command populate-templates

cp -r $PROJECT_DIR/scripts/heudiconv/metadata/stimuli $DATA_DIR/stimuli
cp $PROJECT_DIR/scripts/heudiconv/metadata/*events.json $DATA_DIR/
cp $PROJECT_DIR/scripts/heudiconv/metadata/*bold.json $DATA_DIR/
cp $PROJECT_DIR/scripts/heudiconv/metadata/participants.json $DATA_DIR/
cp $PROJECT_DIR/scripts/heudiconv/metadata/participants.tsv $DATA_DIR/
cp $PROJECT_DIR/scripts/heudiconv/metadata/dataset_description.json $DATA_DIR/
rm -rf $DATA_DIR/sub*/ses-2/fmap/*.bval
rm -rf $DATA_DIR/sub*/ses-2/fmap/*.bvec
rm -rf $DATA_DIR/sub*/ses-1/func/*task-rest*events.tsv
EOF

cat << EOF > heudiconv_fieldmaps.sh
#!/bin/bash
#$ -j y
#$ -l h_vmem=10.0G,s_vmem=9.5G
#$ -o /cbica/projects/GraphLearning/project/logs/\$JOB_NAME.\$JOB_ID

# Assign fieldmaps for all already processed subjects
set -euo pipefail

SCRIPTS_DIR=$PROJECT_DIR/scripts/heudiconv
rm -rf $WORK_DIR/bidsdb

for sub in \`find $DATA_DIR -maxdepth 1 -mindepth 1 -type d -name "sub-*" -printf "%f\n"\`
do
    echo "Subject: \$sub"
    sub=\${sub:4} #take off the sub- part
    for ses in \`find $DATA_DIR/sub-\$sub -maxdepth 1 -mindepth 1 -type d -name "ses-*" -printf "%f\n"\`
    do
        echo "Session: \$ses"
        ses=\${ses:4} #take off the ses- part
        python \${SCRIPTS_DIR}/assign_fieldmaps.py $DATA_DIR \$sub \$ses --overwrite --database_path=$WORK_DIR/bidsdb
    done
done
EOF
