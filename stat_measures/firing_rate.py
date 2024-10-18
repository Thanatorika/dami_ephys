"""Creator: Ren Wang.
Last edited: 2024-Oct-16 by RW.

Code utilities for computing firing rates from spike bins.
"""

import numpy as np

def compute_firing_rate(spike_bins, bin_length):
    # TODO: API

    rate_ratio = 1000 / bin_length

    firing_rate = spike_bins * rate_ratio

    return firing_rate

def compute_firing_rate_mean(firing_rate):
    # TODO: API

    return np.mean(firing_rate, axis=1)