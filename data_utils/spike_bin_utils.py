"""Creator: Ren Wang.
Last edited: 2024-Aug-30 by RW.

Data structure and file utilities for spike bins.
"""

import numpy as np
from . import spike_train_utils as stu

def make_bins_from_trains(spike_trains_dict, bin_length):
    # TODO: API

    num_neurons = stu.get_num_neurons(spike_trains_dict)
    max_spike_time = stu.get_max_spike_time(spike_trains_dict)
    num_bins = int(max_spike_time / bin_length) + 1

    bins = np.zeros((num_neurons, num_bins), dtype=int)

    for neuron_idx in range(num_neurons):
        spike_times = spike_trains_dict[neuron_idx]
        for spike_time in spike_times:
            if spike_time != 0:
                bin_idx = int(spike_time / bin_length)
                bins[neuron_idx, bin_idx] = bins[neuron_idx, bin_idx] + 1

    return bins

def make_bins_from_ks4(spike_clusters, spike_times, bin_length):
    # TODO: API

    spike_trains_dict = stu.make_trains_from_ks4(spike_clusters, spike_times)

    return make_bins_from_trains(spike_trains_dict, bin_length)