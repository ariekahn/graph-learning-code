#!/bin/bash
# Assign fieldmaps for all already processed subjects
set -euo pipefail
PROJECT_DIR=/cbica/projects/GraphLearning/project
DATA_DIR=$PROJECT_DIR/data

# Find each scan

cd /cbica/projects/GraphLearning/project/data/sub-GLS003/ses-1/func

mkdir -p .backup
sub=GLS003
run=1
for scan in sub-${sub}_ses-1_task-graphlearning_run-${run}_bold.nii.gz; do
	# If there isn't already a backup, make one
	if [[ ! -f "./.backup/$scan" ]]; then
		echo "backing up $scan"
		#cp $scan "./.backup/$scan"
	# If there is a backup, restore it
	else
		echo "restoring $scan"
		#cp "./.backup/$scan" $scan
	fi
	#cp "./.backup/
done

# If it's not already backed up, back it up

# 

for sub in `find $DATA_DIR -maxdepth 1 -mindepth 1 -type d -name "sub-*" -printf "%f\n"`
do
    echo "Subject: $sub"
    sub=${sub:4} #take off the sub- part
    python $SCRIPTS_DIR/parse_events.py $DATA_DIR $BEHAVIORAL_DIR $BEHAVIORAL_LIST $sub
done
