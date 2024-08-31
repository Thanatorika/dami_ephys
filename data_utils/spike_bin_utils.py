"""Creator: Ren Wang.
Last edited: 2024-Aug-30 by RW.

Data structure and file utilities for spike bins.
"""

import numpy as np
from . import spike_train_utils as stu

def make_bins_from_trains(spike_trains_dict, bin_length=1):
    # TODO: API

    num_neurons = stu.get_num_neurons(spike_trains_dict)
    max_spike_time = stu.get_max_spike_time(spike_trains_dict)
    num_bins = int(max_spike_time / bin_length) + 1

    spike_bins = np.zeros((num_neurons, num_bins), dtype=int)

    for neuron_idx in range(num_neurons):
        spike_times = spike_trains_dict[neuron_idx]
        for spike_time in spike_times:
            if spike_time != 0:
                bin_idx = int(spike_time / bin_length)
                spike_bins[neuron_idx, bin_idx] = spike_bins[neuron_idx, bin_idx] + 1

    return spike_bins

def get_num_neurons(spike_bins):
    # TODO: API

    return len(spike_bins)


def get_num_bins(spike_bins):
    # TODO: API

    return len(spike_bins[0,:])