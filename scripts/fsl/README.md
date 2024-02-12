Depends on:
- fmriprep

Script order:
1. Prep fmriprep output
    - `make_skullstrip.sh && ./run_all.skullstrip.sh`
    - `make_split.sh && ./run_all.split.sh`
2. Create FSL confound files
    - `make_confounds.sh`
3. Run localizer and level one analysis
    - `make_feat_level-1.sh && ./run_all.feat_level-1.sh`
    - `make_feat_localizer.sh && ./run_all.feat_localizer.sh`
    - `make_feat_learning.sh && ./run_all.feat_learning.sh`
4. Run level two analysis
    - `bypass_registration.sh`
    - `make_feat_level-2.sh && ./run_all.feat_level-2.sh`
5. Mask level two data using localizer
    - `make_loc_masks.sh`
