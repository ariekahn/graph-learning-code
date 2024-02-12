from functools import lru_cache
import pandas as pd
from nilearn.image import load_img, math_img


class Loader:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.fmriprep_dir = f'{project_dir}/derived/fmriprep'
        self.fmriprep_dir = f'{project_dir}/derived/fmriprep'
        self.localizer_dir = f'{project_dir}/derived/feat_localizer'
        self.data_dir = f'{project_dir}/data'

    @lru_cache()
    def load_masked_template(self, subject_id):
        """
        Load the T1w template, and mask it

        derived/fmriprep/sub-{subject_id}/anat/sub-{subject_id}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz

        Usage:
        template = load_masked_template('GLS011')
        """
        # Load T1w template and mask
        template = load_img(f'{self.fmriprep_dir}/sub-{subject_id}/anat/sub-{subject_id}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz')
        template_mask = load_img(f'{self.fmriprep_dir}/sub-{subject_id}/anat/sub-{subject_id}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz')
        template = math_img("img1 * img2", img1=template, img2=template_mask)
        return template

    @lru_cache()
    def load_aparc(self, subject_id):
        """
        Load Freesurfer aparc for localizer

        derived/fmriprep/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz
        """
        aparc = load_img(f'{self.fmriprep_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz')
        return aparc

    @lru_cache()
    def load_schaefer(self, subject_id):
        """
        Load Schaefer BOLD-space
        """
        aparc = load_img(f'{self.project_dir}/derived/schaefer/sub-{subject_id}_Schaefer2018_200Parcels_7Networks_aseg.labelwm-2.nii.gz')
        return aparc

    @lru_cache()
    def load_schaefer_fsaverage(self):
        """
        Load Schaefer BOLD-space
        """
        aparc = load_img(f'{self.project_dir}/derived/freesurfer/fsaverage/mri/Schaefer2018_200Parcels_7Networks_aseg.labelwm-2.nii.gz')
        return aparc

    @lru_cache()
    def load_loc(self, subject_id, hemi='both'):
        """
        Load a mask for LOC

        subject_id : str
            GLSxxx
        hemi : str
            'left', 'right', 'both', or 'separate'
        returns:
            nibabel.nifti1.Nifti1Image

        If separate,
        ctx-lh-lateraloccipital = 1
        ctx-rh-lateraloccipital = 2

        We want 2011 ctx-rh-lateraloccipital and 1011 ctx-lh-lateraloccipital
        """
        roi_str = ''
        if hemi == 'left':
            roi_str = 'img == 1011'
        elif hemi == 'right':
            roi_str = 'img == 2011'
        elif hemi == 'both':
            roi_str = '(img == 1011) | (img == 2011)'
        elif hemi == 'separate':
            roi_str = '(img == 1011) + 2*(img == 2011)'
        else:
            raise ValueError('hemi must be left/right/both/separate')

        aparc = self.load_aparc(subject_id)
        loc = math_img(roi_str, img=aparc)
        return loc

    @lru_cache()
    def load_entorhinal(self, subject_id, hemi='both'):
        """
        Load a mask for Entorhinal Cortex

        subject_id : str
            GLSxxx
        hemi : str
            'left', 'right', 'both', or 'separate'
        returns:
            nibabel.nifti1.Nifti1Image

        If separate,
        ctx-lh-??? = 1
        ctx-rh-??? = 2

        We want 2011 ctx-rh-lateraloccipital and 1011 ctx-lh-lateraloccipital
        """
        roi_str = ''
        if hemi == 'left':
            roi_str = 'img == 1006'
        elif hemi == 'right':
            roi_str = 'img == 2006'
        elif hemi == 'both':
            roi_str = '(img == 1006) | (img == 2006)'
        elif hemi == 'separate':
            roi_str = '(img == 1006) + 2*(img == 2006)'
        else:
            raise ValueError('hemi must be left/right/both/separate')

        aparc = self.load_aparc(subject_id)
        entorhinal = math_img(roi_str, img=aparc)
        return entorhinal

    @lru_cache()
    def load_hippocampus(self, subject_id, hemi='both'):
        """
        Load a mask for Hippocampus

        subject_id : str
            GLSxxx
        hemi : str
            'left', 'right', 'both', or 'separate'
        returns:
            nibabel.nifti1.Nifti1Image

        If separate,
        ctx-lh-??? = 1
        ctx-rh-??? = 2

        We want 2011 ctx-rh-lateraloccipital and 1011 ctx-lh-lateraloccipital
        """
        roi_str = ''
        if hemi == 'left':
            roi_str = 'img == 17'
        elif hemi == 'right':
            roi_str = 'img == 53'
        elif hemi == 'both':
            roi_str = '(img == 17) | (img == 53)'
        elif hemi == 'separate':
            roi_str = '(img == 17) + 2*(img == 53)'
        else:
            raise ValueError('hemi must be left/right/both/separate')

        aparc = self.load_aparc(subject_id)
        hippocampus = math_img(roi_str, img=aparc)
        return hippocampus

    @lru_cache()
    def load_zstat_index(self, subject_id):
        """
        Loads zstat index where thresholded cluster voxels are labeled
        derived/localizer/clusters/sub-{subject_id}/fwhm-5.0/zstat1_index.nii.gz
        """
        img = load_img(f'{self.localizer_dir}/sub-{subject_id}/clusters/fwhm-5.0_zstat1_index.nii.gz')
        return img

    @lru_cache()
    def load_zstat(self, subject_id):
        """
        Loads zstat statistical map
        derived/localizer/zfiles/sub-{subject_id}/fwhm-5.0/zstat1_index.nii.gz
        """
        img = load_img(f'{self.localizer_dir}/sub-{subject_id}/zfiles/fwhm-5.0_zstat1.nii.gz')
        return img

    def load_checks(self, subjects):
        df = pd.DataFrame()
        for subject in subjects:
            for run in range(1, 6):
                events = pd.read_csv(f'{self.data_dir}/sub-{subject}/ses-1/func/sub-{subject}_ses-1_task-graphcheck_run-{run}_events.tsv', sep='\t')
                events['run'] = run
                events['subject'] = subject
                df = pd.concat((df, events))
        df['session'] = 1
        df = df.drop(columns='stim_file').reset_index(drop=True)
        return df

    def load_session_one(self, subjects):
        df = pd.DataFrame()
        for subject in subjects:
            for run in range(1, 6):
                events = pd.read_csv(f'{self.data_dir}/sub-{subject}/ses-1/func/sub-{subject}_ses-1_task-graphlearning_run-{run}_events.tsv', sep='\t')
                events['run'] = run
                events['subject'] = subject
                events['trial'] = range(1, 301)
                events['trial_consecutive'] = events['trial'] + 300 * (run - 1)
                df = pd.concat((df, events))
        df['session'] = 1
        df = df.drop(columns='stim_file').reset_index(drop=True)
        return df

    def load_session_two(self, subjects):
        df = pd.DataFrame()
        for subject in subjects:
            for run in range(1, 9):
                events = pd.read_csv(f'{self.data_dir}/sub-{subject}/ses-2/func/sub-{subject}_ses-2_task-graphrepresentation_run-{run}_events.tsv', sep='\t')
                events['run'] = run
                events['subject'] = subject
                events['trial'] = range(1, 61)
                events['trial_consecutive'] = events['trial'] + 60 * (run - 1)
                df = pd.concat((df, events))
        df['session'] = 2
        df = df.drop(columns='stim_file').reset_index(drop=True)
        return df
