"""Creator: Ren Wang.
Last edited: 2024-Aug-30 by RW.

Data file utilities for spike trains.
"""

import numpy as np

def make_trains_from_ks4(spike_clusters, spike_times):
    # TODO: API

    spike_trains_dict = {}
    for cluster_idx in range(max(spike_clusters)+1):
        spike_times_in_cluster_i = []
        spikes_in_cluster_i = np.argwhere(spike_clusters == cluster_idx)

        for spike in spikes_in_cluster_i:
            spike_time = spike_times[spike[0]]
            spike_times_in_cluster_i.append(spike_time)

        spike_trains_dict[cluster_idx] = spike_times_in_cluster_i

    return spike_trains_dict

def get_num_neurons(spike_trains_dict):
    # TODO: API

    return len(spike_trains_dict)

def get_max_spike_time(spike_trains_dict):
    # TODO: API

    max_list = []
    for x in spike_trains_dict.values():
        max_list.append(max(x))
    
    return max(max_list)

def save_spike_trains(spike_trains_dict, path_to_save):
    # TODO: API

    return
