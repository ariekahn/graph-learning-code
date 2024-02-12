import os


def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return (template, outtype, annotation_classes)


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """
    # Baseline session
    t2w = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_T2w')

    # Fieldmaps
    fmap_pa = create_key(
        '{bids_subject_session_dir}/fmap/{bids_subject_session_prefix}_acq-func_dir-PA_run-1_epi')
    fmap_ap = create_key(
        '{bids_subject_session_dir}/fmap/{bids_subject_session_prefix}_acq-func_dir-AP_run-1_epi')

    # Diffusion Fieldmaps
    dwi_pa = create_key(
        '{bids_subject_session_dir}/fmap/{bids_subject_session_prefix}_acq-dwi_dir-PA_run-1_epi')
    dwi_ap = create_key(
        '{bids_subject_session_dir}/fmap/{bids_subject_session_prefix}_acq-dwi_dir-AP_run-1_epi')

    # graph representation
    graphrepresentation_bold_run1 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-graphrepresentation_run-1_bold')
    graphrepresentation_bold_run2 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-graphrepresentation_run-2_bold')
    graphrepresentation_bold_run3 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-graphrepresentation_run-3_bold')
    graphrepresentation_bold_run4 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-graphrepresentation_run-4_bold')
    graphrepresentation_bold_run5 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-graphrepresentation_run-5_bold')
    graphrepresentation_bold_run6 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-graphrepresentation_run-6_bold')
    graphrepresentation_bold_run7 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-graphrepresentation_run-7_bold')
    graphrepresentation_bold_run8 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-graphrepresentation_run-8_bold')

    # Localizer
    graphlocalizer_bold = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-graphlocalizer_bold')

    # Diffusion
    dwi = create_key(
        '{bids_subject_session_dir}/dwi/{bids_subject_session_prefix}_dwi')


    info = {
        t2w: [],

        dwi: [], dwi_pa: [], dwi_ap: [],

        fmap_pa: [], fmap_ap: [],

        graphrepresentation_bold_run1: [],
        graphrepresentation_bold_run2: [],
        graphrepresentation_bold_run3: [],
        graphrepresentation_bold_run4: [],
        graphrepresentation_bold_run5: [],
        graphrepresentation_bold_run6: [],
        graphrepresentation_bold_run7: [],
        graphrepresentation_bold_run8: [],

        graphlocalizer_bold: [],
    }

    for s in seqinfo:
        protocol = s.protocol_name.lower()

        if ('t2w_spc' in protocol and 'NORM' in s.image_type):
            info[t2w].append(s.series_id)

        elif ('fmri_distortionmap_ap' in protocol):
            info[fmap_ap].append(s.series_id)
        elif ('fmri_distortionmap_pa' in protocol):
            info[fmap_pa].append(s.series_id)

        elif ('graphtask_2_1' in protocol):
            info[graphrepresentation_bold_run1] = [s.series_id]
        elif ('graphtask_2_2' in protocol):
            info[graphrepresentation_bold_run2].append(s.series_id)
        elif ('graphtask_2_3' in protocol):
            info[graphrepresentation_bold_run3].append(s.series_id)
        elif ('graphtask_2_4' in protocol):
            info[graphrepresentation_bold_run4].append(s.series_id)
        elif ('graphtask_2_5' in protocol):
            info[graphrepresentation_bold_run5].append(s.series_id)
        elif ('graphtask_2_6' in protocol):
            info[graphrepresentation_bold_run6].append(s.series_id)
        elif ('graphtask_2_7' in protocol):
            info[graphrepresentation_bold_run7].append(s.series_id)
        elif ('graphtask_2_8' in protocol):
            info[graphrepresentation_bold_run8].append(s.series_id)

        elif ('loc_localizer' in protocol):
            info[graphlocalizer_bold].append(s.series_id)

        elif ('dmri_distortionmap_ap' in protocol):
            info[dwi_ap].append(s.series_id)
        elif ('dmri_distortionmap_pa' in protocol):
            info[dwi_pa].append(s.series_id)
        elif ('dmri' in protocol):
            info[dwi].append(s.series_id)
    return info
