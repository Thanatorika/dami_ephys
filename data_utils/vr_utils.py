"""Creator: Ren Wang.
Last edited: 2024-Aug-19 by RW.

Stimuli utilities for Kilosort4.
"""

import numpy as np

# Section I: Information Retrieval
## Get the information about VR stimuli and object movement from log files.

def load_vr_log(filepath):
    # TODO: API
    # head_yaw,head_thrust,head_slip,current_condition,loom_cur_distance,ms_now

    vr_log = np.loadtxt(filepath, dtype=str)
    vr_log_matrix = np.array([x.split(',') for x in vr_log], dtype=str)

    return vr_log_matrix

def get_head_yaw(vr_log_matrix):
    # TODO: API

    return vr_log_matrix[:,0].astype(int)

def get_head_thrust(vr_log_matrix):
    # TODO: API

    return vr_log_matrix[:,1].astype(int)

def get_head_slip(vr_log_matrix):
    # TODO: API

    return vr_log_matrix[:,2].astype(int)

def get_current_condition(vr_log_matrix):
    # TODO: API

    return vr_log_matrix[:,3].astype(int)

def get_loom_cur_distance(vr_log_matrix):
    # TODO: API

    return vr_log_matrix[:,4].astype(float)

def get_ms_now(vr_log_matrix):
    # TODO: API

    return vr_log_matrix[:,5].astype(int)

# Section II: Stimuli Information
## Form a matrix of stimuli related information, form stimuli windows, and mark spikes by windows.

def get_loom_idx(stimuli_info, stimulus_type, round_idx):
    # TODO:API

    pos =  np.where((stimuli_info[0] == round_idx+1) & (stimuli_info[1] == stimulus_type))

    return pos[0][0]

def get_loom_time(stimuli_info, stimulus_type, round_idx):
    # TODO:API

    pos =  np.where((stimuli_info[0] == round_idx+1) & (stimuli_info[1] == stimulus_type))

    return stimuli_info[2, pos[0][0]]

def get_loom_times_by_stimuli(stimuli_info, stimulus_type):
    # TODO:API

    indices = np.argwhere(stimuli_info[1] == stimulus_type)

    return stimuli_info[2][indices].flatten()

def get_loom_times_by_round(stimuli_info, round_idx):
    # TODO:API

    indices = np.argwhere(stimuli_info[0] == round_idx)

    return stimuli_info[2][indices].flatten()

def mark_spikes_by_window(stimuli_info, spike_times, pre_window, post_window):
    # TODO: API

    times = np.copy(spike_times)
    spike_window_stimuli = np.ones_like(times) * -1
    spike_window_round = np.ones_like(times) * -1

    num_windows = len(stimuli_info[0])

    for window_idx in range(num_windows):
        loom_time = stimuli_info[2, window_idx]
        start_time = loom_time - pre_window
        end_time = loom_time + post_window

        round_idx = stimuli_info[0, window_idx] - 1
        stimulus_type = stimuli_info[1, window_idx]

        indices = (times > start_time) & (times <= end_time)
        
        spike_window_stimuli[indices] = stimulus_type
        spike_window_round[indices] = round_idx
    
    return spike_window_stimuli.astype(int), spike_window_round.astype(int)

def times_in_windows(stimuli_info, spike_times, pre_window, post_window):
    # TODO: API

    times = np.copy(spike_times)
    num_windows = len(stimuli_info[0])

    for window_idx in range(num_windows):
        loom_time = stimuli_info[2, window_idx]
        start_time = loom_time - pre_window
        end_time = loom_time + post_window

        indices = (times > start_time) & (times <= end_time)
        times[indices] = times[indices] - loom_time

    return times.flatten()

# Section III: Object Movement
## Compute the movement information of the experiment object