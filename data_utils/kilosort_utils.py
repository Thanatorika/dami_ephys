"""Creator: Ren Wang.
Last edited: 2024-Aug-29 by RW.

Data file utilities for Kilosort4 (KS4 in short).

Some of the API documents are generated with Bito AI and GPT-4o mini, and are labeled with [AI generated].
These API docs will be human proofread and modified if necessary. 
"""

import numpy as np
import sys
import importlib.util

# Section 1: Information Retrieval
# Get the data and parameters from the KS4 output files.
def load_sample_rate(ks4_directory):
    """Load the sample rate of the KS4 run.

    Code from https://stackoverflow.com/questions/19009932/import-arbitrary-python-source-file-python-3-3,
    answer by Stefan Scherfke to import the `.py` file.

    Parameters
    ----------
    ks4_directory: string
        the directory of the KS4 output files.

    Returns
    ----------
    sample_rate (float): the sampling rate of the KS4 outputs.
    """

    try:
        file_path = f'{ks4_directory}/params.py'

        spec = importlib.util.spec_from_file_location('params', file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['params'] = module
        spec.loader.exec_module(module)
    except FileNotFoundError:
        print("Kilosort4 directory not found!")
        return
    
    return module.sample_rate

def load_clusters(ks4_directory):
    """Load spike train from KS4 output directory.

    Parameters
    ----------
    ks4_directory (string): the directory of the KS4 output files.

    Returns
    ----------
    clusters (numpy array): an array with length of the number of spikes in the whole KS4 output,
                            whose values are the cluster indices of the spikes.
    """

    try:
        file_path = f'{ks4_directory}/spike_clusters.npy'

        clusters = np.load(file_path)
    except FileNotFoundError:
        print("Kilosort4 directory not found!")
        return

    return clusters

def load_times(ks4_directory):
    """Load spike train from KS4 output directory.

    Parameters
    ----------
    ks4_directory (string): the directory of the KS4 output files.

    Returns
    ----------
    times (numpy array): an array with length of the number of spikes in the whole KS4 output,
                         whose values are the timestamps of the spikes.
    """

    try:
        sample_rate = load_sample_rate(ks4_directory)

        file_path = f'{ks4_directory}/spike_times.npy'

        times = np.load(file_path)
        times = times / sample_rate * 1000
    except FileNotFoundError:
        print("Kilosort4 directory not found!")
        return

    return times

def get_cluster_number(ks4_directory):
    """Get the number of clusters in the KS4 output.

    Parameters
    ----------
    ks4_directory (string): the directory of the KS4 output files.

    Returns
    ----------
    max_clusters (int): the number of spike clusters in the KS4 output.
    """

    clusters = load_clusters(ks4_directory)
    return max(clusters) + 1

def get_max_time(ks4_directory):
    """Get the time of the last spike in the KS4 output.

    Parameters
    ----------
    ks4_directory (string): the directory of the KS4 output files.

    Returns
    ----------
    max_time (float): the maximum time point in the KS4 output.
    """

    times = load_times(ks4_directory)
    return max(times)

# Section 2: CLuster Filtering
# Make filtered and reduced spike cluster ids and spike times from a list of selected cluster ids to keep or dump.

def make_keep_filter(ks4_spike_clusters, cluster_indices_to_filter):
    """Make a filter of which clusters to keep from all clusters output by KS4.

    Parameters
    ----------
    ks4_spike_clusters (numpy array): the content of the `spike_clusters.npy` file in KS4 output directory.
    cluster_indices_to_filter (numpy array): an array of the indices to keep from the numpy arrays in KS4 directory.

    Returns
    ----------
    cluster_filter (numpy array): a filter of which indices of the spike clusters and spike times from KS4 to keep.
                                  please apply the filter on the numpy arrays from the KS4 directory.
                                  Example: filtered_spike_times = raw_spike_times[cluster_filter].
    """

    cluster_filter = []
    for spike_cluster in ks4_spike_clusters:
        if spike_cluster in cluster_indices_to_filter:
            cluster_filter.append(True)
        else:
            cluster_filter.append(False)
    
    return cluster_filter

def make_dump_filter(ks4_spike_clusters, cluster_indices_to_filter):
    """Make a filter of which clusters to delete from all clusters output by KS4.

    Parameters
    ----------
    ks4_spike_clusters (numpy array): the content of the `spike_clusters.npy` file in KS4 output directory.
    cluster_indices_to_filter (numpy array): an array of the indices to delete from the numpy arrays in KS4 directory.

    Returns
    ----------
    cluster_filter (numpy array): a filter of which indices of the spike clusters and spike times from KS4 to keep.
                                  please apply the filter on the numpy arrays from the KS4 directory.
                                  Example: filtered_spike_times = raw_spike_times[cluster_filter].
    """

    keep_filter = make_keep_filter(ks4_spike_clusters, cluster_indices_to_filter)

    return np.logical_not(keep_filter)

def make_filtered_clusters_and_times(ks4_spike_clusters, ks4_spike_times, cluster_filter):
    """[AI generated] This function filters spike clusters and spike times based on a given filter condition.

    Parameters
    ----------
    ks4_spike_clusters  (array-like): The original spike cluster data, typically an array or list containing identifiers for different clusters.
    ks4_spike_times  (array-like): The original spike time data, typically an array or list containing timestamps for the corresponding spikes.
    cluster_filter  (array-like): The filter condition, usually a boolean array or index array indicating which clusters and times to retain.

    Returns
    ------
    Filtered spike clusters (array-like): The spike cluster array filtered according to  cluster_filter.
    Filtered spike times (array-like): The spike time array filtered according to  cluster_filter.
    """
    
    return ks4_spike_clusters[cluster_filter], ks4_spike_times[cluster_filter]

def reduce_clusters_from_filtered(filtered_clusters):
    # TODO: API

    sort_order = np.argsort(filtered_clusters)
    reverse_order = np.argsort(sort_order)

    unique_clusters, counts= np.unique(filtered_clusters, return_counts=True)
    num_unique_clusters = len(unique_clusters)

    reduced_clusters_sorted = []
    for idx in range(num_unique_clusters):
        count = counts[idx]
        for cnt in range(count):
            reduced_clusters_sorted.append(idx)

    reduced_clusters = np.array(reduced_clusters_sorted)[reverse_order]

    return reduced_clusters

def make_reduced_clusters_and_times(ks4_spike_clusters, ks4_spike_times, cluster_filter):
    # TODO: API

    filtered_clusters, filtered_times = make_filtered_clusters_and_times(ks4_spike_clusters, ks4_spike_times, cluster_filter)

    reduced_clusters = reduce_clusters_from_filtered(filtered_clusters)

    return reduced_clusters, filtered_times

# Section 3: Spikes Segmentation
# Take a time or spike count interval from the given spike clusters and spike times  

def time_segmentation(spike_clusters, spike_times, start_time, end_time):
    # TODO: API

    segment_filter = (spike_times > start_time) & (spike_times <= end_time)
    segmented_clusters = spike_clusters[segment_filter]
    segmented_times = spike_times[segment_filter]

    return segmented_clusters, segmented_times

def time_segmentation_second(spike_clusters, spike_times, start_time, end_time):
    # TODO: API

    return time_segmentation(spike_clusters, spike_times, start_time * 1000, end_time * 1000)

def time_segmentation_reduced(spike_clusters, spike_times, start_time, end_time):
    """Same as time_segmentation, but the spike times will be reduced to times from the segmentation start time.

    Parameters
    ----------
    spike_clusters (numpy array): the spike clusters of the spike train to be segmented.
    spike_times (numpy array): the spike times of the spike train to be segmented.
    start_time (float): the start time of the segmentation interval.
    end_time (float): the end time of the segmentation interval.

    Returns
    ----------
    reduced_clusters (numpy array): the spike clusters of the segmented spikes. 
    reduced_times (numpy array): the passed time of the segmented spikes from the given start time.
    """

    segmented_clusters, segmented_times = time_segmentation(spike_clusters, spike_times, start_time, end_time)

    return segmented_clusters, segmented_times - start_time

def spike_count_segmentation(spike_clusters, spike_times, start_count, end_count):
    # TODO: API

    return spike_clusters[start_count : end_count], spike_times[start_count : end_count]

def spike_count_segmentation_reduced(spike_clusters, spike_times, start_count, end_count):
    # TODO: API
    
    segmented_clusters, segmented_times = spike_count_segmentation(spike_clusters, spike_times, start_count, end_count)

    return segmented_clusters, segmented_times - spike_times[start_count]