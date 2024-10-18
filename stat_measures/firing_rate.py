"""Creator: Ren Wang.
Last edited: 2024-Oct-16 by RW.

Code utilities for computing firing rates from spike bins.
"""

def compute_firing_rate(spike_bins, bin_length):
    # TODO: API

    rate_ratio = 1000 / bin_length

    firing_rate = spike_bins * rate_ratio

    return firing_rate

