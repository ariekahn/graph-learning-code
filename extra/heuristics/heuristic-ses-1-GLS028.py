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
    t1w = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_T1w')

    # Fieldmaps
    fmap_pa_one = create_key(
        '{bids_subject_session_dir}/fmap/{bids_subject_session_prefix}_acq-func_dir-PA_run-1_epi')
    fmap_ap_one = create_key(
        '{bids_subject_session_dir}/fmap/{bids_subject_session_prefix}_acq-func_dir-AP_run-1_epi')
    fmap_pa_two = create_key(
        '{bids_subject_session_dir}/fmap/{bids_subject_session_prefix}_acq-func_dir-PA_run-2_epi')
    fmap_ap_two = create_key(
        '{bids_subject_session_dir}/fmap/{bids_subject_session_prefix}_acq-func_dir-AP_run-2_epi')

    # Rest
    rest_bold_run1 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-rest_run-1_bold')
    rest_bold_run2 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-rest_run-2_bold')
    rest_bold_run3 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-rest_run-3_bold')
    rest_bold_run4 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-rest_run-4_bold')

    # graph learning
    graphlearning_bold_run1 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-graphlearning_run-1_bold')
    graphlearning_bold_run2 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-graphlearning_run-2_bold')
    graphlearning_bold_run3 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-graphlearning_run-3_bold')
    graphlearning_bold_run4 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-graphlearning_run-4_bold')
    graphlearning_bold_run5 = create_key(
        '{bids_subject_session_dir}/func/{bids_subject_session_prefix}_task-graphlearning_run-5_bold')


    info = {
        t1w: [],

        fmap_pa_one: [], fmap_ap_one: [],
        fmap_pa_two: [], fmap_ap_two: [],

        rest_bold_run1: [],
        rest_bold_run2: [],
        rest_bold_run3: [],
        rest_bold_run4: [],

        graphlearning_bold_run1: [],
        graphlearning_bold_run2: [],
        graphlearning_bold_run3: [],
        graphlearning_bold_run4: [],
        graphlearning_bold_run5: [],
    }

    for s in seqinfo:
        protocol = s.protocol_name.lower()

        if ('mprage_1mm' in protocol and s.dim3 == 176):
            info[t1w].append(s.series_id)

        elif ('rest_1_1' in protocol):
            info[rest_bold_run1].append(s.series_id)
        elif ('rest_1_2' in protocol):
            info[rest_bold_run2].append(s.series_id)
        elif ('rest_1_3' in protocol):
            info[rest_bold_run3].append(s.series_id)
        elif ('rest_1_4' in protocol):
            info[rest_bold_run4].append(s.series_id)

        elif ('fmri_distortionmap_ap' in protocol):
            if len(info[fmap_ap_one]) == 0:
                info[fmap_ap_one].append(s.series_id)
            else:
                info[fmap_ap_two].append(s.series_id)
        elif ('fmri_distortionmap_pa' in protocol):
            if len(info[fmap_pa_one]) == 0:
                info[fmap_pa_one].append(s.series_id)
            else:
                info[fmap_pa_two].append(s.series_id)

        elif ('graphtask_1_1' in protocol):
            info[graphlearning_bold_run1].append(s.series_id)
        elif ('graphtask_1_2' in protocol):
            info[graphlearning_bold_run2].append(s.series_id)
        elif ('graphtask_1_3' in protocol):
            info[graphlearning_bold_run3].append(s.series_id)
        elif ('graphtask_1_4' in protocol):
            info[graphlearning_bold_run4].append(s.series_id)
        elif ('graphtask_1_5' in protocol):
            info[graphlearning_bold_run5].append(s.series_id)

    return info
