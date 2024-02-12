import pandas as pd
import os
import os.path as op
import sys
import argparse
import logging

# ex: python parse_events.py ../../data ../../behavioral ../../extra/behavioral_list.tsv GLS011
parser = argparse.ArgumentParser('Create BIDS-structured event data.')
parser.add_argument('bids_dir', type=str, help='BIDS directory base')
parser.add_argument('behavioral_dir', type=str, help='Behavioral record directory base')
parser.add_argument('behavioral_list', type=str, help='tsv file containing subjects and runs')
parser.add_argument('subject_id', type=str, help='Subject (without sub- prefix)')
parser.add_argument('--dry_run', action='store_true', help="Don't write changes.")
parser.add_argument('-v', '--verbose', action='store_true', help='Show intended changes')
args = parser.parse_args()

subject_id = args.subject_id
bids_dir = op.abspath(args.bids_dir)
behavioral_dir = op.abspath(args.behavioral_dir)
behavioral_list = args.behavioral_list
dry_run = args.dry_run

graphs = {'GLS003': 'Modular', 'GLS004': 'Modular', 'GLS005': 'Lattice', 'GLS006': 'Lattice',
          'GLS008': 'Modular', 'GLS009': 'Lattice', 'GLS010': 'Modular', 'GLS011': 'Modular',
          'GLS013': 'Modular', 'GLS014': 'Lattice', 'GLS016': 'Modular', 'GLS017': 'Lattice',
          'GLS018': 'Modular', 'GLS019': 'Modular', 'GLS020': 'Lattice', 'GLS021': 'Lattice',
          'GLS022': 'Modular', 'GLS023': 'Modular', 'GLS024': 'Lattice', 'GLS025': 'Modular',
          'GLS026': 'Modular', 'GLS027': 'Lattice', 'GLS028': 'Lattice', 'GLS030': 'Lattice',
          'GLS033': 'Lattice', 'GLS034': 'Lattice', 'GLS036': 'Modular', 'GLS037': 'Modular',
          'GLS038': 'Modular', 'GLS039': 'Modular', 'GLS040': 'Lattice', 'GLS041': 'Lattice',
          'GLS043': 'Lattice', 'GLS044': 'Modular', 'GLS045': 'Lattice',
          }

if args.verbose:
    logging.basicConfig(level=logging.INFO)

logging.info(f'Subject: {subject_id}')
logging.info(f'BIDS Dir: {bids_dir}')
logging.info(f'Behavioral Dir: {behavioral_dir}')

subject_df = pd.read_csv(behavioral_list, sep='\t')
subject = subject_df[subject_df['subject'] == subject_id]
if len(subject) == 0:
    logging.error(f'No subject found: {subject_id}')
    sys.exit(1)
elif len(subject) > 1:
    logging.error(f'Multiple subjects found: {subject_id}')
    sys.exit(1)
subject = subject.iloc[0]


def load_session_one(subject):
    scan = op.abspath(f'{behavioral_dir}/{subject.subject}/{subject.scan_one_fixed}.csv')
    data = pd.read_csv(scan)
    return data


def load_session_two(subject):
    scan = op.abspath(f'{behavioral_dir}/{subject.subject}/{subject.scan_two_fixed}.csv')
    data = pd.read_csv(scan)
    return data


session_one = load_session_one(subject)
session_two = load_session_two(subject)


def get_block_start(scan):
    return scan['block.timings.start'].iloc[0]


def bidsify_learning(scan):
    onset = (scan['trial.timings.stimulus'] - scan['block.timings.start']).round(3)
    duration = scan['trial.timings.rt'].round(3)
    response_time = scan['trial.timings.rt'].round(3)
    node = scan['node'].astype(int)
    shape = scan['shape'].astype(int) + 1
    trial_type = 'node_' + node.astype(str)
    variation = scan['variation'].astype(int)
    stim_file = ['images/shapes_120/img_{}_{}_0.png'.format(s, v) for s, v in zip(shape, variation)]
    correct = scan['trial.correct'].astype(bool)
    movement = scan['trial.response'].str.replace(' ', '')
    movement_correct = '[' + \
                       scan['button_1'].astype(int).astype(str) + \
                       scan['button_2'].astype(int).astype(str) + \
                       scan['button_3'].astype(int).astype(str) + \
                       scan['button_4'].astype(int).astype(str) + \
                       scan['button_5'].astype(int).astype(str) + \
                       ']'
   first_movement = scan['trial.first-response'].str.replace(' ', '')

    events = pd.DataFrame(dict(
        onset=onset,
        duration=duration,
        trial_type=trial_type,
        response_time=response_time,
        stim_file=stim_file,
        node=node,
        shape=shape,
        variation=variation,
        correct=correct,
        movement=movement,
        movement_correct=movement_correct,
        first_movement=first_movement,
        graph=graphs[subject_id]
    ))

    events = events.fillna('n/a')
    return events


def bidsify_check(scan, offset):
    onset = (scan['time_stim_start'] + offset).round(3)
    duration = scan['time_stim_end'] - scan['time_stim_start']
    response_time = scan['trial.timings.rt'].round(3)
    node = scan['node'].astype(int)
    shape = scan['shape'].astype(int) + 1
    trial_type = 'node_' + node.astype(str)
    variation = scan['variation'].astype(int)
    is_hamiltonian = scan['is_hamiltonian_walk'].astype(bool)
    stim_file = ['images/shapes_300/img_{}_{}_0.png'.format(s, v) for s, v in zip(shape, variation)]
    correct = scan['trial.correct'].astype(bool)
    movement = scan['trial.response'].str.replace(' ', '')
    movement_correct = '[' + \
                       scan['button_1'].astype(int).astype(str) + \
                       scan['button_2'].astype(int).astype(str) + \
                       scan['button_3'].astype(int).astype(str) + \
                       scan['button_4'].astype(int).astype(str) + \
                       scan['button_5'].astype(int).astype(str) + \
                       ']'

    events = pd.DataFrame(dict(
        onset=onset,
        duration=duration,
        trial_type=trial_type,
        response_time=response_time,
        stim_file=stim_file,
        node=node,
        shape=shape,
        variation=variation,
        correct=correct,
        is_hamiltonian=is_hamiltonian,
        movement=movement,
        movement_correct=movement_correct,
        graph=graphs[subject_id]
    ))

    events = events.fillna('n/a')
    return events


def bidsify_test(scan):
    onset = scan['time_stim_start']
    duration = scan['time_stim_end'] - scan['time_stim_start']
    response_time = scan['trial.timings.rt'].round(3)
    node = scan['node'].astype(int)
    shape = scan['shape'].astype(int) + 1
    trial_type = 'node_' + node.astype(str)
    variation = scan['variation'].astype(int)
    is_hamiltonian = scan['is_hamiltonian_walk'].astype(bool)
    stim_file = ['images/shapes_300/img_{}_{}_0.png'.format(s, v) for s, v in zip(shape, variation)]
    correct = scan['trial.correct'].astype(bool)
    movement = scan['trial.response'].str.replace(' ', '')
    movement_correct = '[' + \
                       scan['button_1'].astype(int).astype(str) + \
                       scan['button_2'].astype(int).astype(str) + \
                       scan['button_3'].astype(int).astype(str) + \
                       scan['button_4'].astype(int).astype(str) + \
                       scan['button_5'].astype(int).astype(str) + \
                       ']'

    events = pd.DataFrame(dict(
        onset=onset,
        duration=duration,
        trial_type=trial_type,
        response_time=response_time,
        stim_file=stim_file,
        node=node,
        shape=shape,
        variation=variation,
        correct=correct,
        is_hamiltonian=is_hamiltonian,
        movement=movement,
        movement_correct=movement_correct,
        graph=graphs[subject_id]
    ))

    events = events.fillna('n/a')
    return events


def bidsify_localizer(scan):
    onset = scan['time_stim_start']
    duration = (scan['time_stim_end'] - scan['time_stim_start']).round(3)
    trial_type = scan['trial_type']
    node = scan['node'].astype(int)
    shape = scan['shape'].astype(int) + 1
    variation = scan['variation'].astype(int)
    stim_file = ['images/localizer_{}/img_{}_{}.png'.format(t, s, v) for t, s, v in zip(trial_type, shape, variation)]

    events = pd.DataFrame(dict(
        onset=onset,
        duration=duration,
        stim_file=stim_file,
        trial_type=trial_type,
        node=node,
        shape=shape,
        variation=variation,
        graph=graphs[subject_id]
    ))
    events.loc[events['trial_type'] == 'blank',
               ['node', 'shape', 'variation', 'stim_file']] = 'n/a'
    events = events.fillna('n/a')
    return events


sessions_learning = [
    'learning_0',
    'learning_1',
    'learning_2',
    'learning_3',
    'learning_4',
]

sessions_check = [
    'check_0',
    'check_1',
    'check_2',
    'check_3',
    'check_4',
]

sessions_test = [
    'test_0',
    'test_1',
    'test_2',
    'test_3',
    'test_4',
    'test_5',
    'test_6',
    'test_7',
]

# Keep track of when the learning blocks start, so we can correct timing
# for onsets in the check blocks
learning_block_starts = []
for i, ses_name in enumerate(sessions_learning):
    ses = session_one.loc[(session_one['Participant ID'] == subject_id)
                          & (session_one['block.name'] == ses_name)]
    learning_block_starts.append(get_block_start(ses))
    bids_events = bidsify_learning(ses)
    os.makedirs(f'{bids_dir}/sub-{subject_id}/ses-1/func/', exist_ok=True)
    filename = f'{bids_dir}/sub-{subject_id}/ses-1/func/sub-{subject_id}_ses-1_task-graphlearning_run-{i+1}_events.tsv'
    if not dry_run:
        bids_events.to_csv(filename, sep='\t', index=False)

for i, ses_name in enumerate(sessions_check):
    ses = session_one.loc[(session_one['Participant ID'] == subject_id)
                          & (session_one['block.name'] == ses_name)]
    check_block_start = get_block_start(ses) - learning_block_starts[i]
    bids_events = bidsify_check(ses, check_block_start)
    os.makedirs(f'{bids_dir}/sub-{subject_id}/ses-1/func/', exist_ok=True)
    filename = f'{bids_dir}/sub-{subject_id}/ses-1/func/sub-{subject_id}_ses-1_task-graphcheck_run-{i+1}_events.tsv'
    if not dry_run:
        bids_events.to_csv(filename, sep='\t', index=False)

for i, ses_name in enumerate(sessions_test):
    ses = session_two.loc[(session_two['Participant ID'] == subject_id)
                          & (session_two['block.name'] == ses_name)]
    bids_events = bidsify_test(ses)
    os.makedirs(f'{bids_dir}/sub-{subject_id}/ses-2/func/', exist_ok=True)
    filename = f'{bids_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphrepresentation_run-{i+1}_events.tsv'
    if not dry_run:
        bids_events.to_csv(filename, sep='\t', index=False)

ses = session_two.loc[(session_two['Participant ID'] == subject_id)
                      & (session_two['block.type'] == 'localizer')]
bids_events = bidsify_localizer(ses)
filename = f'{bids_dir}/sub-{subject_id}/ses-2/func/sub-{subject_id}_ses-2_task-graphlocalizer_events.tsv'
if not dry_run:
    bids_events.to_csv(filename, sep='\t', index=False)
