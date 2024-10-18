"""Creator: Ren Wang.
Last edited: 2024-Oct-18 by RW.

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

def compute_firing_rate_variance(firing_rate):
    # TODO: API

    return np.var(firing_rate, axis=1)

def compute_firing_rate_standard_deviation(firing_rate):
    # TODO: API

    return np.std(firing_rate, axis=1)

def compute_firing_rate_coefficient_of_variance(firing_rate):
    # TODO: API

    fr_mean = compute_firing_rate_mean(firing_rate)
    fr_std = compute_firing_rate_standard_deviation(firing_rate)

    return fr_mean / fr_std