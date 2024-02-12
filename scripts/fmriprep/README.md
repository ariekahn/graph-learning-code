# fMRIPrep and related scripts

`make_fmriprep.sh` generates the runfile for fmriprep.

There are two utility scripts to summarize some fmriprep stats:
1. `compile_fd.sh` reports framewise displacement for each subject
2. `make_outlying_trs.sh` creates a report on which TRs appear to be outliers, particularly for flagging non-steady-state frames
