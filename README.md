# Graph Learning Data Processing

General outline:

## Scripts and configuration

```
.
├── config.cfg - Edit this to point to the top-level project directory
├── scripts/ - All processing scripts
│   └── scripts/singularity/ - scripts to build singularity containers
├── notebooks/ - Notebooks plotting results used in paper
├── singularity/ - singularity containers
├── extra/ - configuration and subject lists
│   ├── behavioral\_list.txt - Used by `parse_events.py`.
│   ├── heuristics\_list.txt - list of heuristic files to use for each subject. Used by `heudiconv`
│   ├── heuristics/
│   │   ├── heuristic-ses-1.py
│   │   ├── heuristic-ses-2.py
│   |   └── heuristic-ses-1-GLSxxx.py - for any custom heuristics
│   ├── license.txt - Freesurfer license file
│   ├── scan\_list.tsv - used by `import` scripts. Contains 
│   ├── temp/ - Directory for storing temporary xcpengine configuration files
│   └── xcp\_designs/ - xcpEngine design files
├── behavioral/ - Raw behavior data
├── dicom/ - Raw dicom data
├── data/ - BIDS unprocessed data
├── derived/ - all processing results
```

### scan\_list.txt
TSV formatted as:
```
SUBJECT SCAN_1  SCAN_2
GLSXXX  20190101.XXX 20190101.XXX
```

### behavioral\_list.txt
TSV formatted as:
```
subject training    scan_one    scan_one_fixed  scan_two    scan_two_fixed
GLSXXX  GLSXXX_0_TwoDay_2019_May_1_1700    GLSXXX_1_TwoDay_2019_May_30_1824    GLSXXX_1_TwoDay_2019_May_30_1824_fixed  GLSXXX_2_TwoDay_2019_May_31_1900_fixed  GLSXXX_2_TwoDay_2019_May_31_1900_fixed
```

### heuristics\_list.txt
TSV formatted as:
```
Subject heuristic_1 heuristic_2
GLSXXX  heuristic-ses-1.py  heuristic-ses-2-GLSXXX.py
```
# Instructions

## Configuration
Configuration Steps:
1. edit `config.cfg`
2. place freesurfer license at `extra/license.txt`
3. Build any necessary singularity containers: e.g. `qsub scripts/singularity/build_fmriprep_image.sh`

## Importing
Importing dicoms:
1. `scripts/import/import_from_rico`
2. `scripts/import/sort_dicoms`
3. `scripts/import/compress_dicoms`

Converting to nii and creating BIDS datastructure
1. `scripts/heudiconv/make_heudiconv.sh && scripts/heudiconv/run_all.heudiconv.sh`
2. `scripts/behavioral/parse_events.sh`

## Preprocessing
fMRIPrep:
1. `scripts/fmriprep`

FSL FEAT Processing:
1. `scripts/mvpa/make_feat_localizer.sh && scripts/mvpa/run_all.feat_localizer.sh`
2. `scripts/mvpa/make_feat_representation_lss.sh && scripts/mvpa/run_all.feat_representation_lss.sh`

## Processing
### Behavior and MVPA Classification
1. `scripts/mvpa/make_mvpa_classifier.sh && scripts/mvpa/run_all.mvpa_classifier.sh`
2. `notebooks/Behavior_and_MVPA_Classification.ipynb`

### RDMs
1. `scripts/rsa/make_crossval_euclidean_mats.sh && scripts/rsa/run_all.crossval_euclidean_mats.sh`
2. `notebooks/Cross-validated_Euclidean_RDMs.ipynb`

### Whole-Brain RDM Lower Bounds
1. `scripts/rsa/make_pymvpa_searchlight.sh && scripts/rsa/run_all.pymvpa_searchlight.sh`
2. `scripts/rsa/make_rsa_lower_bound_pymvpa.sh && scripts/rsa/run_all.rsa_lower_bound_pymvpa.sh`
3. `notebooks/RDM_wholebrain_lower_bounds.ipynb`

### Within-subject Pattern Consistency
First, you'll need to install CoSMoMVPA: <http://cosmomvpa.org>, then pre-compile the MATLAB script
1. Install CoSMoMVPA: <http://cosmomvpa.org/get_started.html>, making sure it's accessible from MATLAB
2. `cd scripts/rsa && mcc -m cosmo_make_rdm_searchlight_per_run.m && cd ../..`
3. `scripts/rsa/make_cosmo_make_rdm_searchlight_per_run.sh && cd scripts/rsa && scripts/rsa/run_all.cosmo_make_rdm_searchlight_per_run.sh`
4. `scripts/rsa/make_rsa_consistency.sh && scripts/rsa/run_all.rsa_consistency.sh`

### Dimensionality
1. `scripts/dimensionality/make_dimensionality.sh && scripts/dimensionality/run_all.dimensionality.sh`
2. `notebooks/Dimensionality.ipynb`

### Power analysis
1. `scripts/poweranalysis/make_poweranalysis.sh && scripts/poweranalysis/run_all.poweranalysis.sh`

## Behavioral Discrimination
1. See `notebooks/discrimination_analysis/analysis.Rmd`