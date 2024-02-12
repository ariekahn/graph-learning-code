# Graph Learning Data Processing

## Export from Flywheel
`fw export bids data -p GraphLearning`

## Subjects with issues

### Summary
The following subjects have data that will need custom BIDS curation. In a number of these cases we had to restart and ended up with dupliate scan names. A couple may or may not be valid data, but most should be minor fixes.
- GLS003
    - Shimming was broken on second day of scanning, had to use manual shimming. Patient also needed to exit the scanner after the fourth graph task, and re-entered afterwards. Graph task 5 on day two may not have been triggered correctly and may be off by a TR. Due to concerns over manual shimming, ran the T2 at the very start of the scan.
    - Solution: Have two distortion maps, have to apply second map to rest of scans
- GLS008
    - Not sure...
    - Have duplicate of graphtask_2_1, disregard first
- GLS014
    - Had to position eyes below the eye holes in head coil due to hair. Initially was blocked by eye holes, re-adjusted after the first 5min RS and reran the whole sequence.
    - Same scanning issues as scan 1. Tanya had to reposition a couple times.
    - Scan 1: Disregard first distortion map and rest
    - Scan 2: we ran the localizer a bunch but shouldn't be a problem
- GLS020
    - Duplicate nii of BOLD_graphtask_2_1, might be a conversion problem
- GLS025
    - Two T2 files: Did it use different coils????
    - Seems like it might have just converted the files wrong - issue with dcm2niix?
    - Solution: Batch reconverted, telling it to combine all 2d slices
- GLS026
- GLS028
    - Timed out on graphtask_1_1. Had to use restroom after graphtask_1_2, reran autoalign afterwards. Also may have missed first TR in graphtask_2_2, check how many are recorded.
- GLS030
- GLS033
    - Scanner stopped midway through graphtask_2_2. Restarted from beginning.
- GLS034
    - Had to rerun localizer
- GLS026
- 

### Clean subjects

- sub-GLS004
- sub-GLS005
- sub-GLS006
- sub-GLS009
- sub-GLS010
- sub-GLS011
- sub-GLS013
- sub-GLS016
- sub-GLS017
- sub-GLS018
- sub-GLS019
- sub-GLS021
- sub-GLS022
- sub-GLS023
- sub-GLS024
- sub-GLS027
- sub-GLS044

## fMRI Prep

Building singularity container:

https://fmriprep.readthedocs.io/en/stable/installation.html

```bash
singularity build $HOME/singularity/fmriprep-1.5.0.simg docker://poldracklab/fmriprep:1.5.0
```

Now in `singularity/fmriprep-1.5.0.simg`

```bash
singularity run --cleanenv ~/singularity/fmriprep-1.5.0.simg --participant_label sub-GLS013 data fmriprep participant
```
