# -*- coding: utf-8 -*-
# System packages
import os

# Numerical packages
import numpy as np
import pandas as pd
from scipy import interpolate

# Graphical packages
from matplotlib import pyplot as plt

# Local repository
from Code.general_configs import GC
from Code.GUIs.GUI_configs import default_colors
from Code.IO_operations.files_and_folders import read_info_file
from Code.IO_operations.spikes_hdf5 import read_hdf5_data, read_hdf5_OSSA_results, read_sorting_summary_table
from Code.Workflows.Spike_sorting_workflow.IO_operations import read_spikes_txt
from Code.Workflows.Spike_sorting_workflow.OSSA.visualization import set_figure_style


################################################################################
# Correlograms
################################################################################
def correlogram(spike_times, spike_clusters=None, sample_rate=40000., bin_size_ms=2., window_size_ms=500., normalize_area=True):
    """
    Compute all pairwise cross-correlograms among the clusters of spike_times.
    Abbreviations: ACG autocorrelogram, CCG crosscorrelogram, CG correlogram

    :param spike_times: [numpy-array] Timestamps of each spike (in seconds).
    :param spike_clusters: [numpy-array] ID of each spike.
    :param sample_rate: [float] Sampling rate of signal.
    :param bin_size_ms: [float] Bin size in ms for correlogram histogram.
    :param window_size_ms: [float] Maximum lag at which to compute correlation.
    :param normalize_area: [bool] If True, normalize the area to 1.

    :return correlograms: the correlogram matrix, containing auto- and cross-
        correlograms.
    :return ref2asy: It contains the ratio between refractory and asymptotic
        correlations.
    """
    # Transform all inputs to float numbers
    sample_rate = float(sample_rate)
    bin_size = bin_size_ms / 1000.  # in s
    window_size_ms /= 1000.  # in s

    # Calculate CG parameters
    binsize = int(bin_size * sample_rate)  # in samples
    if binsize < 1:
        binsize = 1
    winsize_bins = 2 * int(window_size_ms / bin_size) + 1

    # If there are no spikes, build an empty correlogram
    if spike_times is None or spike_times.shape[0] == 0:
        spike_times = np.array([0], dtype=int)

    if not hasattr(spike_clusters, "__iter__"):
        spike_clusters = np.ones_like(spike_times)
    # Get the continuous cluster index
    cluster_idx, spike_clusters_i, n_spikes_cluster = np.unique(spike_clusters, return_inverse=True, return_counts=True)
    n_clusters = len(cluster_idx)

    # Sort the spike times
    if n_clusters == 1:
        spike_times.sort()  # spike_clusters_i contains only one value
    else:
        array = pd.DataFrame(columns=["time","id"])
        array["time"] = spike_times
        array["id"] = spike_clusters_i
        array.sort_values(by="time", inplace=True)
        spike_times = array["time"].values.astype(float)
        spike_clusters_i = array["id"].values.astype(int)

    # Convert spike times to samples
    spike_samples = np.multiply(spike_times, sample_rate).astype(int)

    # At a given shift, the mask precises which spikes have matching spikes within the correlogram time window
    mask = np.ones_like(spike_samples, dtype=np.bool)

    # Pre-allocate output
    correlograms = np.zeros((n_clusters, n_clusters, winsize_bins // 2 + 1), dtype=np.float32)

    # The loop continues as long as there is at least one spike with a matching spike
    n_spikes = len(spike_samples)
    shift = 1
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift
        spike_diff = spike_samples[shift:] - spike_samples[:n_spikes - shift]

        # Bin the delays between spike i and spike i+shift
        spike_diff_b = spike_diff // binsize

        # Spikes with no matching spikes are masked.
        mask[:-shift][spike_diff_b > (winsize_bins // 2)] = False

        # Cache the masked spike delays
        m = mask[:-shift].copy()
        d = spike_diff_b[m]

        # Find the indices in the raveled correlograms array that need to be
        # incremented, taking into account the spike clusters
        indices = np.ravel_multi_index((spike_clusters_i[:-shift][m], spike_clusters_i[+shift:][m], d), correlograms.shape)

        # Increment the matching spikes in the correlograms array
        _increment(correlograms.ravel(), indices)
        shift += 1

    # Remove ACG peak at 0-lag
    correlograms[np.arange(n_clusters), np.arange(n_clusters), 0] = 0

    # Copy to obtain symmetric correlogram
    correlograms = _symmetrize_correlograms(correlograms)

    # Normalize correlogram to the height of 0-lag
    for row in range(n_clusters):
        for col in range(n_clusters):
            correlograms[row, col, :] /= np.sqrt(
                n_spikes_cluster[row] * n_spikes_cluster[col])

    # Normalize to have area = 1
    if normalize_area:
        for row in range(n_clusters):
            for col in range(n_clusters):
                norm_factor = np.sum(correlograms[row, col, :])
                if norm_factor > 0:
                    correlograms[row, col, :] /= norm_factor
    return correlograms

def _increment(arr, indices):
    """Increment some indices in a 1D vector of non-negative integers.
    Repeated indices are taken into account."""
    bbins = np.bincount(indices)
    arr[:len(bbins)] += bbins
    return arr

def _symmetrize_correlograms(correlograms):
    """Return the symmetrized version of the CCG arrays."""

    n_clusters, _, n_bins = correlograms.shape
    assert n_clusters == _

    # We symmetrize c[i, j, 0].
    # This is necessary because the algorithm in correlograms()
    # is sensitive to the order of identical spikes.
    correlograms[..., 0] = np.maximum(correlograms[..., 0], correlograms[..., 0].T)

    sym = correlograms[..., 1:][..., ::-1]
    sym = np.transpose(sym, (1, 0, 2))

    return np.dstack((sym, correlograms))

def correlogram_make_lag_axis(window_size_ms, bin_size_ms):
    lag = np.arange(-window_size_ms, window_size_ms + bin_size_ms, bin_size_ms)
    return lag


################################################################################
# Helper functions
################################################################################
def divide0(a, b, rep):
    """Divide two numbers but replace its result if division is not
    possible, e.g., when dividing a number by 0.

    :param a: [numeric] A number.
    :param b: [numeric] A number.
    :param rep: [numeric] If a/b is not defined return this number instead.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = rep
    return c


def compute_array_range(values, padding=0):
    """Compute the range of an array, and optionally add some padding."""
    if not isinstance(values, np.ndarray):
        try:
            values = np.hstack(values).ravel()
        except TypeError:
            return
    if values.shape[0] == 0:  # no elements
        return np.array([-.0001, +.0001], dtype=values.dtype)
    if values.shape[0] == 1:  # 1 element
        return np.array([values-.0001, values+.0001], dtype=values.dtype)
    else:
        values_extrema = [values.min(), values.max()]
        if isinstance(padding, bool):
            if padding is True:
                padding = 0.025
            elif padding is False:
                padding = 0
        if padding > 0:
            values_range = values_extrema[1] - values_extrema[0]
            padding = values_range * padding
            values_extrema[0] -= padding
            values_extrema[1] += padding
        return np.array(values_extrema, dtype=values.dtype)


def compute_CCG_all_units(spikes_filename, info_filename, channel_names=None, write_spyking_circus=False, read_from_text_file=False, reset_cluster_id=False, print_pdf=False):
    """Compute cross-correlograms between channels."""
    # Initialize variables
    sorted_with_ROSS = False
    DATA_TABLE = None
    SORTING_RESULTS = None
    OSSA_results = None

    # Read .info file
    INFO = read_info_file(info_filename)
    sampling_rate = INFO["sampling rate"]

    # Check which channels have spikes
    if read_from_text_file:
        # Load text file
        DATA_TABLE = read_spikes_txt(spikes_filename)
        SORTING_RESULTS = INFO["sorting results"]

        # Channels with spikes are those that have at least one spike in the table
        channels_with_spikes = list(set(DATA_TABLE["channel"]))

    else:
        # Read sessions_with_spikes and OSSA_results from from .hdf5 file
        OSSA_results, channels_with_spikes = read_hdf5_OSSA_results(spikes_filename)

        if write_spyking_circus:  # In this case, ignore the sessions_with_spikes from processing data with OSSA
            channels_with_spikes = list()
            all_channel_names = INFO["channel names"]
            for ch_name in all_channel_names:
                # Check the size of the table
                table_size = read_hdf5_data(spikes_filename, ch_name, column_list="cluster_id_auto").shape[0]
                if table_size > 0:
                    channels_with_spikes.append(ch_name)

    # If user chose some spikes, take only that subset
    if channel_names is None:  # Take all spikes
        channel_names = list(channels_with_spikes)  # This makes a copy, just to be sure
    else:
        # Keep only the channels that both the user requested AND have spikes on them
        channel_names = list(set(channel_names).intersection(set(channels_with_spikes)))

    # Sort channel names
    channel_names.sort()

    # Make filename containing the recording name and the selected channels
    base_filename = os.path.splitext(os.path.splitext(spikes_filename)[0])[0]
    channels = ".CCG(" + "_".join(channel_names) + ")"
    fname = base_filename + channels + ".pdf"

    # Check that we can write the pdf, if requested
    if print_pdf:
        # Before starting check whether we can write the file
        try:
            f = open(fname, "w")
            f.close()
        except IOError as e:
            if e.errno == 13:  # "[Errno 13] Permission denied: ..."
                msg = "WARNING: Cannot access file '%s' because it is open in another process" % fname
                raise msg
            else:  # Raise error normally
                raise e

    # Get correlogram's bin size and window width from the general_configs
    CCG_bin_size = GC.update_NE_unit_data_correlogram_bin_width
    CCG_max_lag = GC.update_NE_unit_data_correlogram_window_width

    # Get the name of the column from which will read data in the hdf5 file
    if write_spyking_circus:
        column_to_read = "cluster_id_auto"
    else:
        column_to_read = "cluster_id_manual"

    # Pre-allocate table to hold cluster ids and labels
    TABLE = pd.DataFrame(columns=["channel_name", "cluster_id", "color", "cluster_label"])
    COLORS = default_colors()
    for ch_name in channel_names:
        if read_from_text_file:
            unit_ids = np.unique(DATA_TABLE.loc[DATA_TABLE["channel"] == ch_name, "cluster_id"])
            unit_types = SORTING_RESULTS.loc[SORTING_RESULTS["channel_name"] == ch_name, "cluster_type"].values
            SUMMARY_TABLE = pd.DataFrame(columns=["id", "cluster_type"])
            SUMMARY_TABLE["id"] = unit_ids
            SUMMARY_TABLE["cluster_type"] = unit_types
        else:
            if not write_spyking_circus:  # Read sorting summary from the hdf5 file
                SUMMARY_TABLE = read_sorting_summary_table(spikes_filename, ch_name, ignore_errors=False)
                # Remove noisy cluster
                SUMMARY_TABLE.drop(np.where(SUMMARY_TABLE["cluster_type"] == "n")[0], inplace=True)
                SUMMARY_TABLE.reset_index(inplace=True, drop=True)

            else:  # Make a summary table
                # Get a list of clusters for this channel
                clusters_ids = np.unique(read_hdf5_data(spikes_filename, ch_name, column_list="cluster_id_auto"))
                # Make table
                SUMMARY_TABLE = pd.DataFrame(columns=["id", "cluster_type"])
                SUMMARY_TABLE["id"] = clusters_ids
                SUMMARY_TABLE["cluster_type"] = "m"

        # Make a new color for each channel
        channel_color = next(COLORS)

        # Check whether we have processed this channel with OSSA
        if not read_from_text_file and not write_spyking_circus:
            sorted_with_ROSS = OSSA_results.loc[OSSA_results["channel"] == ch_name, "SpikeSorter"].values[0]

        # Get info
        for row in range(SUMMARY_TABLE.shape[0]):
            # Get cluster type
            cluster_type = SUMMARY_TABLE.loc[row, "cluster_type"]
            if cluster_type == "m":
                cluster_type = "MU"
            elif cluster_type == "s":
                cluster_type = "SU"
            # Add the prefix "unsorted" if skipping manual sorting
            if write_spyking_circus:
                cluster_label_prefix = "u"
            else:
                # Adapt label according to whether we have processed the data with OSSA
                if sorted_with_ROSS:
                    cluster_label_prefix = ""
                else:
                    cluster_label_prefix = "u"

            # Get cluster_id
            cluster_id = SUMMARY_TABLE.loc[row, "id"]

            # Make cluster label (which includes the channel name)
            cluster_label = ch_name + "_" + cluster_label_prefix + cluster_type
            if reset_cluster_id:
                cluster_label += str("%02i" % (row+1))
            else:
                cluster_label += str("%02i" % cluster_id)

            # Concatenate data
            table_row = TABLE.shape[0]
            TABLE.set_value(table_row, "channel_name", ch_name)
            TABLE.set_value(table_row, "cluster_id", cluster_id)
            TABLE.set_value(table_row, "color", channel_color)
            TABLE.set_value(table_row, "cluster_label", cluster_label)

    # Collect the spike train of each cluster
    SPIKE_TRAINS = np.empty((0,), dtype=float)
    SPIKE_ID = np.empty_like(SPIKE_TRAINS, dtype=int)
    for row in range(TABLE.shape[0]):
        # Get where the data should be read
        ch_name = str(TABLE.loc[row, "channel_name"])
        cluster_id = int(TABLE.loc[row, "cluster_id"])

        # Get spike timestamps and convert them to seconds
        if read_from_text_file:
            timestamps = DATA_TABLE.loc[(DATA_TABLE["channel"] == ch_name) & (DATA_TABLE["cluster_id"] == cluster_id), "time"].values
        else:
            timestamps = read_hdf5_data(spikes_filename, ch_name, where="%s == %i" % (column_to_read, cluster_id), column_list="timestamp")
            timestamps = timestamps.astype(np.float32) / sampling_rate
        # Make an array marking the cluster id
        spike_id = np.ones_like(timestamps, dtype=int) * row  # Use 'row' as counter
        # Keep data in memory
        SPIKE_TRAINS = np.concatenate((SPIKE_TRAINS, timestamps))
        SPIKE_ID = np.concatenate((SPIKE_ID, spike_id))

    # Pre-allocate DataFrame to hold the data in memory
    CCG = correlogram(SPIKE_TRAINS, SPIKE_ID, sampling_rate, CCG_bin_size, CCG_max_lag)
    # Look at how many subplots to make
    n_clusters = CCG.shape[0]
    n_bins = CCG.shape[2]

    # Open figure and make a grid of subplots
    style_dict = set_figure_style((.96, .96, .96))
    title_color = style_dict["axes.labelcolor"]
    fig, ax = plt.subplots(nrows=n_clusters, ncols=n_clusters, sharex="all")
    if "AxesSubplot" in str(type(ax)):  # ax is not a numpy array
        ax = np.atleast_2d(np.array([ax]))
    ax = np.fliplr(ax)  # So ACG are on anti-diagonal
    # Start hiding all axes
    [ax[i, j].set_visible(False) for i in np.arange(n_clusters) for j in np.arange(n_clusters)]
    # Get the x-values of the histograms
    x_data = np.linspace(-CCG_max_lag, CCG_max_lag, n_bins)

    # Plot CCGs only once, in the upper-triangle of the array. Find
    # indices of these plots
    ccg_idx_row, ccg_idx_col = np.triu_indices(n_clusters)
    cluster_pairs = list(zip(ccg_idx_row.tolist(), ccg_idx_col.tolist()))

    # Go over each pair of cross-correlation histograms, ...
    for cluster_piar in range(len(cluster_pairs)):
        i = cluster_pairs[cluster_piar][0]
        j = cluster_pairs[cluster_piar][1]
        # Get CCG data
        data = CCG[i, j, :]
        # Get channel color for ACG; use black for CCGs
        color = TABLE.loc[i, "color"] if i == j else np.zeros(3)

        # Draw bars
        ax[i, j].set_visible(True)
        ax[i, j].bar(left=x_data-CCG_bin_size/2., height=data, width=CCG_bin_size, bottom=0, color=color, edgecolor="None", linewidth=0.0)

        # Draw a red line at lag 0
        ax[i, j].axvline(0, color="r", lw=.5)

        # Add the cluster label above the first row and on the left of the
        # first column
        if i == 0:
            ax[i, j].set_title(TABLE.loc[j, "cluster_label"], fontdict={"fontsize": 14}, color=title_color)
        if j == n_clusters - 1:
            ax[i, j].set_ylabel(TABLE.loc[i, "cluster_label"], fontdict={"fontsize": 14})
        # Add the cluster label beside each ACG
        if i==j and i!=0 and j!=0 and i!=n_clusters-1 and j!=n_clusters-1:
            ax[i, j].set_ylabel(TABLE.loc[i, "cluster_label"], fontdict={"fontsize": 14})
            ax[i, j].yaxis.set_label_position("right")
            ax[i, j].yaxis.label.set_rotation(270)
            ax[i, j].yaxis.label.set_verticalalignment("bottom")

        # Skip fixing the axes limits if there is no data available
        if np.max(data) > 0:
            # Fix y-lims
            y_range = [0, compute_array_range(data, padding=True)[1]]
            ax[i, j].set_ylim(y_range)

        # Remove all y-tick labels but leave the grid on
        ax[i, j].set_yticks(list())
        ax[i, j].set_yticklabels(list())

    # Restore the visibility of the x-axes in ACGs (plt.subplots() hides them when axes limits are shared)
    for i in range(n_clusters):
        for label in ax[i, i].get_xticklabels():
            label.set_visible(True)
        ax[i, i].xaxis.offsetText.set_visible(True)
        ax[i, i].tick_params(axis='x', colors=title_color)

    # Fix axes appearance
    ax[-1, -1].set_xlabel("lag (ms)")
    ax[0, 0].set_xlim(-CCG_max_lag, CCG_max_lag)

    # Remove unused axes and empty space
    ax = ax.ravel().tolist()
    empty_axes_idx = np.where([not ax[i].get_visible() for i in range(len(ax))])[0]
    [fig.delaxes(ax[i]) for i in empty_axes_idx]

    # Show figure if user requested it, otherwise return figure handle
    if print_pdf:
        # Make figure bigger
        fig.set_size_inches(n_clusters * 3.5, n_clusters * 3.5)
        # Apply tight layout
        fig.tight_layout(pad=0.1, h_pad=1, w_pad=1, rect=[.05, .05, .95, .95])
        # Print to pdf
        fig.savefig(fname)
        # Close figure
        plt.close(fig)

    else:
        # Maximize figure
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # Apply tight layout
        fig.tight_layout(pad=0, h_pad=0, w_pad=.5, rect=[.01, .01, .99, .99])
        # Return handle
        return fig


def compute_average_waveform(spikes_hdf5_filename, info_filename, root_data_dir, write_spyking_circus):
    """

    :param spikes_hdf5_filename:
    :param info_filename:
    :param root_data_dir:
    :param write_spyking_circus:
    :return:
    """
    # Initialize empty variables to silence code inspections (PEP8)
    cluster_label = ""

    # Get the date_compound_tested
    date_compound_tested = os.path.split(root_data_dir)[1]

    # Read info from .info file
    INFO = read_info_file(info_filename)
    sampling_rate = INFO["sampling rate"]
    recording_duration = INFO["recording duration"]

    # Get names of channels with spikes
    if not write_spyking_circus:  # Read sessions_with_spikes from OSSA_results in the hdf5 file
        channels_with_spikes = read_hdf5_OSSA_results(spikes_hdf5_filename)[1]

    else:
        channels_with_spikes = list()
        all_channel_names = INFO["channel names"]
        for ch_name in all_channel_names:
            # Check the size of the table
            table_size = read_hdf5_data(spikes_hdf5_filename, ch_name, column_list="cluster_id_auto").shape[0]
            if table_size > 0:
                channels_with_spikes.append(ch_name)

    # Make a list from the channels with spikes
    channel_names = list(channels_with_spikes)

    # Make filename containing the recording name and the selected channels
    base_filename = os.path.splitext(os.path.splitext(spikes_hdf5_filename)[0])[0]
    channels = ".waveforms(" + "_".join(channel_names) + ")"
    fname = base_filename + channels + ".pdf"

    # Check that we can write the pdf
    try:
        f = open(fname, "w")
        f.close()
    except IOError as e:
        if e.errno == 13:  # "[Errno 13] Permission denied: ..."
            msg = "WARNING: Cannot access file '%s' because it is open in another process" % fname
            raise msg
        else:  # Raise error normally
            raise e

    # Make path of all .waveforms files
    waveforms_filename = [os.path.join(root_data_dir, date_compound_tested + "_" + i + ".waveforms") for i in channel_names]

    # Read waveform duration from general_configs
    spike_waveform_duration_ms = GC.spike_waveform_duration
    # Get intervals before / after peak
    waveform_before_peak_ms = np.abs(spike_waveform_duration_ms[0])
    waveform_after_peak_ms = np.abs(spike_waveform_duration_ms[1])
    waveform_before_peak = int(waveform_before_peak_ms / 1000. * sampling_rate)
    waveform_after_peak = int(waveform_after_peak_ms / 1000. * sampling_rate)
    # Waveform duration in samples
    spike_waveform_duration = waveform_before_peak + waveform_after_peak + 1
    # Make x-axis for plots
    waveform_plot_x = np.arange(-waveform_before_peak, waveform_after_peak + 1)
    # Get x-axis limits
    time_axis_waveform_minmax = (waveform_plot_x.min(), waveform_plot_x.max())
    # Make x-ticks and -labels
    waveform_plot_xticks = np.arange(-waveform_before_peak_ms, waveform_after_peak_ms + .001, 0.5)
    waveform_plot_xticklabels = np.array(["%1.1f" % i for i in waveform_plot_xticks])
    # Convert from time to samples
    waveform_plot_x_waveform_ms = waveform_plot_x / sampling_rate * 1000.
    waveform_plot_xticks = waveform_plot_x[[np.argmin(np.abs(waveform_plot_x_waveform_ms - i)) for i in waveform_plot_xticks]]

    # Map in memory the waveform files
    WAVEFORMS = memmap_waveform_file(waveforms_filename, spikes_hdf5_filename, spike_waveform_duration, channel_names)

    # Get a list of good cluster IDs in the whole recording
    n_units = 0
    for ch_name in channel_names:
        if not write_spyking_circus:  # Read summary table
            SUMMARY_TABLE = read_sorting_summary_table(spikes_hdf5_filename, ch_name, ignore_errors=False)
            # Discard units labeled as "noise"
            SUMMARY_TABLE.drop(np.where(SUMMARY_TABLE["cluster_type"] == "n")[0], axis=0, inplace=True)

            # Add number of units to the total
            n_units += SUMMARY_TABLE.shape[0]

        else:
            clusters_ids = np.unique(read_hdf5_data(spikes_hdf5_filename, ch_name, column_list="cluster_id_auto"))
            n_units += clusters_ids.shape[0]

    # Given this number of units, make a figure with enough subplots
    n_rows_plots = int(np.ceil(np.sqrt(n_units)))
    n_cols_plots = int(np.ceil(n_units / float(n_rows_plots)))

    # Make styles and colors
    style_dict = set_figure_style((.96, .96, .96))
    title_color = style_dict["axes.labelcolor"]
    COLORS = default_colors()  # plot colors
    # Open figure and make a grid of subplots
    fig, ax = plt.subplots(nrows=n_rows_plots, ncols=n_cols_plots, sharex="all")
    if "AxesSubplot" in str(type(ax)):  # ax is not a numpy array
        ax = np.atleast_2d(np.array([ax]))
    # Make ax a linear array
    ax = ax.ravel()
    # Initialize a counter for the current axes
    current_axes_idx = 0
    # Make an array to mark which axes belong to which channel
    ax_channel = np.zeros_like(ax, dtype=int) - 1

    # Get the name of the column from which will read data in the hdf5 file
    if write_spyking_circus:
        column_to_read = "cluster_id_auto"
    else:
        column_to_read = "cluster_id_manual"

    # Initialize variable to store y-axis limits
    Y_LIMITS = [list() for _ in range(len(channel_names))]

    # Loop through each channel
    for ch_idx, ch_name in enumerate(channel_names):
        # Read data from the hdf5 file
        data = read_hdf5_data(spikes_hdf5_filename, table_name=ch_name, column_list=column_to_read)

        if not write_spyking_circus:  # Read summary table
            SUMMARY_TABLE = read_sorting_summary_table(spikes_hdf5_filename, ch_name, ignore_errors=False)
            # Discard units labeled as "noise"
            SUMMARY_TABLE.drop(np.where(SUMMARY_TABLE["cluster_type"] == "n")[0], inplace=True)
            SUMMARY_TABLE.reset_index(drop=True, inplace=True)
        else:
            # Read the array of cluster ids
            clusters_id_array = read_hdf5_data(spikes_hdf5_filename, ch_name, column_list="cluster_id_auto")
            clusters_ids = np.unique(clusters_id_array)
            # Allocate empty table
            SUMMARY_TABLE = pd.DataFrame(columns=["id", "cluster_type", "FR"])
            for row, unit_id in enumerate(clusters_ids):
                SUMMARY_TABLE.set_value(row, "id", unit_id)
                SUMMARY_TABLE.set_value(row, "cluster_type", "m")
                SUMMARY_TABLE.set_value(row, "FR", np.where(clusters_id_array==unit_id)[0].shape[0] / recording_duration)

        # Get the id of these units
        units = np.unique(SUMMARY_TABLE["id"])

        # Make a color for this channel
        channel_color = COLORS.next()

        # Loop through units
        for unit_counter, unit_id in enumerate(units):
            # Get the indices of the waveforms where we have this unit
            wave_idx = np.where(data == unit_id)[0]

            # Read the waveforms, and compute mean and standard deviation
            waveforms = WAVEFORMS[ch_idx][wave_idx, :]
            mean_waveform = np.mean(waveforms, axis=0)
            sd_waveform = np.std(waveforms, axis=0)

            # Plot shaded area of mean+-sd
            ax[current_axes_idx].fill_between(waveform_plot_x,
                                              mean_waveform-sd_waveform, mean_waveform+sd_waveform,
                                              facecolor=channel_color, edgecolor="None",
                                              alpha=.5)
            # Plot mean waveform as a black, thick line
            ax[current_axes_idx].plot(waveform_plot_x, mean_waveform, color="k", lw=2.5)

            # Add the prefix "unsorted" if skipping manual sorting
            if write_spyking_circus:
                cluster_label_prefix = "u"
            else:
                cluster_label_prefix = ""
            # Make title based on channel name and unit type
            unit_type = SUMMARY_TABLE.loc[SUMMARY_TABLE["id"] == unit_id, "cluster_type"].values[0]
            if unit_type == "m":
                cluster_label = "MU"
            elif unit_type == "s":
                cluster_label = "SU"
            title = ch_name + "_" + cluster_label_prefix + cluster_label + str("%02i" % (unit_counter + 1))
            # Add title to axes
            ax[current_axes_idx].set_title(title, color=title_color)

            # Store current y-axis limits
            plotted_data = np.concatenate((mean_waveform-sd_waveform, mean_waveform+sd_waveform)).ravel()
            Y_LIMITS[ch_idx].append([plotted_data.min(), plotted_data.max()])

            # Write FR of the current unit in the top-right corner of the plot
            FR = SUMMARY_TABLE.loc[SUMMARY_TABLE["id"] == unit_id, "FR"].values[0]
            if FR > 0.1:
                FR_string = "%.1f Hz" % FR
            else:
                FR_string = "<0.1 Hz"
            ax[current_axes_idx].annotate(FR_string, xy=(.99, .99), xytext=(.99, .99), xycoords="axes fraction", ha="right", va="top", clip_on=True, zorder=100)

            # Mark the current axes to a channel
            ax_channel[current_axes_idx] = ch_idx

            # Increase counter for current axes
            current_axes_idx += 1

    # Hide axes that don't display data
    [i.set_visible(False) for i in ax[n_units:]]

    # Fix appearance of all axes for the same channel
    for ch_idx in range(len(channel_names)):
        # Get the axes
        these_axes = ax[ax_channel==ch_idx]

        # Get y-limits for this channel
        ylims = np.array(Y_LIMITS[ch_idx])
        ylims = [np.min(ylims[:, 0]), np.max(ylims[:, 1])]
        # Set y-limits
        [i.set_ylim(ylims[0], ylims[1]) for i in these_axes]

        # Hide y-ticks on all axes except the first one
        [i.set_yticks([]) for i in these_axes[1:]]

        # Make y-ticks and include 0
        yticks_step = np.abs(ylims).min() * .99
        negative_steps = np.arange(0, ylims[0], -yticks_step)
        positive_steps = np.arange(0, ylims[1], yticks_step)
        yticks = np.unique(np.concatenate((negative_steps, positive_steps)))
        yticklabels = ["%.2f" % i for i in yticks]
        # Set ticks and labels
        [i.set_yticks(yticks) for i in these_axes]
        [i.set_yticklabels([]) for i in these_axes]
        these_axes[0].set_yticklabels(yticklabels)

    # Change color of all ticks
    [i.tick_params(axis='both', colors=title_color) for i in ax]

    # Add x-labels to last row
    [i.set_xlabel("time (ms)") for i in ax[-n_cols_plots:]]
    # Add y-label to first plot only
    ax[0].set_ylabel("amplitude (mV)")

    # Fix x-axis (limits and labels)
    ax[0].set_xlim(time_axis_waveform_minmax)
    ax[0].set_xticks(waveform_plot_xticks)
    ax[0].set_xticklabels(waveform_plot_xticklabels)

    # Change font-size of all x- and y-ticks
    [[t.label.set_fontsize(12) for t in i.xaxis.get_major_ticks()] for i in ax]
    [[t.label.set_fontsize(12) for t in i.yaxis.get_major_ticks()] for i in ax]
    [i.xaxis.label.set_size(14) for i in ax]
    [i.yaxis.label.set_size(14) for i in ax]
    [i.title.set_size(14) for i in ax]

    # Remove unused axes and empty space
    empty_axes_idx = np.where([not ax[i].get_visible() for i in range(ax.shape[0])])[0]
    [fig.delaxes(ax[i]) for i in empty_axes_idx]
    # Make figure bigger
    # fig.set_size_inches(n_cols_plots * 3.5, n_rows_plots * 3.5)
    # Apply  tight layout
    fig.tight_layout(pad=0.1, h_pad=.5, w_pad=.5, rect=[0, 0, 1, 1])

    # Print to pdf
    fig.savefig(fname)
    # Close figure
    plt.close(fig)


def compute_waveform_features(spikes_hdf5_filename, info_filename, root_data_dir, write_spyking_circus):
    """

    :param spikes_hdf5_filename:
    :param info_filename:
    :param root_data_dir:
    :param write_spyking_circus:
    :return:
    """
    # Initialize output variable
    SUMMARIZED_RESULTS = pd.DataFrame(columns=["name", "unit_type", "good", "peak_FWHM_ms", "trough_FWHM_ms", "peak_to_trough_ms", "peak_amplitude_mV", "trough_amplitude_mV", "FR"])

    # Get the date_compound_tested
    date_compound_tested = os.path.split(root_data_dir)[1]

    # Read info from .info file
    INFO = read_info_file(info_filename)
    sampling_rate = INFO["sampling rate"]
    recording_duration = INFO["recording duration"]

    # Get names of channels with spikes
    if not write_spyking_circus:  # Read sessions_with_spikes from OSSA_results in the hdf5 file
        channels_with_spikes = read_hdf5_OSSA_results(spikes_hdf5_filename)[1]

    else:
        channels_with_spikes = list()
        all_channel_names = INFO["channel names"]
        for ch_name in all_channel_names:
            # Check the size of the table
            table_size = read_hdf5_data(spikes_hdf5_filename, ch_name, column_list="cluster_id_auto").shape[0]
            if table_size > 0:
                channels_with_spikes.append(ch_name)

    # Make a list from the channels with spikes
    channel_names = list(channels_with_spikes)

    # Make filename containing the recording name and the selected channels
    base_filename = os.path.splitext(os.path.splitext(spikes_hdf5_filename)[0])[0]
    fname = base_filename + ".unit_summarized_results.txt"

    # Make path of all .waveforms files
    waveforms_filename = [os.path.join(root_data_dir, date_compound_tested + "_" + i + ".waveforms") for i in channel_names]

    # Read the waveform duration from the general_configs
    # Read waveform duration from general_configs
    spike_waveform_duration_ms = GC.spike_waveform_duration
    # Get intervals before / after peak
    waveform_before_peak_ms = np.abs(spike_waveform_duration_ms[0])
    waveform_after_peak_ms = np.abs(spike_waveform_duration_ms[1])
    waveform_before_peak = int(waveform_before_peak_ms / 1000. * sampling_rate)
    waveform_after_peak = int(waveform_after_peak_ms / 1000. * sampling_rate)
    # Waveform duration in samples
    spike_waveform_duration = waveform_before_peak + waveform_after_peak + 1
    # Waveform axes
    waveform_x_original = np.arange(spike_waveform_duration)
    # Up-sample x-axis
    upsampling_factor = 20
    waveform_x_upsampled = np.linspace(waveform_x_original[0], waveform_x_original[-1], waveform_x_original.shape[0]*upsampling_factor)  # upsampling factor is 20
    # Get peak position
    peak_position_original = waveform_before_peak
    peak_position_upsampled = np.argmin(np.abs(waveform_x_upsampled-peak_position_original))  # closest point to original value

    # Map in memory the waveform files
    WAVEFORMS = memmap_waveform_file(waveforms_filename, spikes_hdf5_filename, spike_waveform_duration, channel_names)

    # Get the name of the column from which will read data in the hdf5 file
    if write_spyking_circus:
        column_to_read = "cluster_id_auto"
    else:
        column_to_read = "cluster_id_manual"

    # Loop through each channel
    for ch_idx, ch_name in enumerate(channel_names):
        # Read data from the hdf5 file
        data = read_hdf5_data(spikes_hdf5_filename, table_name=ch_name, column_list=column_to_read)

        if not write_spyking_circus:  # Read summary table
            SUMMARY_TABLE = read_sorting_summary_table(spikes_hdf5_filename, ch_name, ignore_errors=False)
            # Discard units labeled as "noise"
            SUMMARY_TABLE.drop(np.where(SUMMARY_TABLE["cluster_type"] == "n")[0], axis=0, inplace=True)
            SUMMARY_TABLE.reset_index(drop=True, inplace=True)

        else:
            # Read the array of cluster ids
            clusters_id_array = read_hdf5_data(spikes_hdf5_filename, ch_name, column_list="cluster_id_auto")
            clusters_ids = np.unique(clusters_id_array)
            # Allocate empty table
            SUMMARY_TABLE = pd.DataFrame(columns=["id", "cluster_type", "FR"])
            for row, unit_id in enumerate(clusters_ids):
                SUMMARY_TABLE.set_value(row, "id", unit_id)
                SUMMARY_TABLE.set_value(row, "cluster_type", "m")
                SUMMARY_TABLE.set_value(row, "n_spikes", np.where(clusters_id_array==unit_id)[0].shape[0])

        # Get the id of these units
        units = np.unique(SUMMARY_TABLE["id"])

        # Loop through units
        for unit_counter, unit_id in enumerate(units):
            # Get the indices of the waveforms where we have this unit
            wave_idx = np.where(data == unit_id)[0]

            # Read the waveforms, and compute mean and standard deviation
            waveforms = WAVEFORMS[ch_idx][wave_idx, :]
            mean_waveform = np.mean(waveforms, axis=0)

            # Check whether peak is positive or negative
            peak_polarity = np.sign(mean_waveform[peak_position_original])

            # Make the peak always positive
            mean_waveform *= peak_polarity

            # Apply spline interpolation to upsample the waveform
            tck = interpolate.splrep(waveform_x_original, mean_waveform, s=0)
            mean_waveform_upsampled = interpolate.splev(waveform_x_upsampled, tck, der=0)

            # Compute the 1st derivative of the mean waveform
            derivative = np.diff(mean_waveform_upsampled)
            # The trough occurs at 0-velocity after the peak
            trough_position_upsampled = np.argmax(np.sign(derivative[peak_position_upsampled:])) + peak_position_upsampled

            # Compute full width at half maximum (FWHM) for both prominences
            FWHMp = _compute_FWHM(mean_waveform_upsampled, peak_position_upsampled)
            FWHMt = _compute_FWHM(mean_waveform_upsampled * -1, trough_position_upsampled)

            # Down-sample and convert to ms
            FWHMp = (FWHMp / float(upsampling_factor)) / sampling_rate * 1000.
            FWHMt = (FWHMt / float(upsampling_factor)) / sampling_rate * 1000.

            # Calculate the delay between peak and trough, in ms
            P2T = (np.abs(peak_position_upsampled - trough_position_upsampled) / float(upsampling_factor))  / sampling_rate * 1000.

            # Get peak and trough amplitudes
            peak_amplitude = mean_waveform_upsampled[peak_position_upsampled]
            trough_amplitude = mean_waveform_upsampled[trough_position_upsampled]

            # Get FR and unit type from summary table
            FR = SUMMARY_TABLE.loc[SUMMARY_TABLE["id"] == unit_id, "n_spikes"].values[0] / recording_duration
            unit_type = SUMMARY_TABLE.loc[SUMMARY_TABLE["id"] == unit_id, "cluster_type"].values[0]

            # Add the prefix "unsorted" if skipping manual sorting
            if write_spyking_circus:
                cluster_label_prefix = "u"
            else:
                cluster_label_prefix = ""
            # Make unit label based on channel name and unit type
            if unit_type == "m":
                unit_type = "MU"
            elif unit_type == "s":
                unit_type = "SU"
            unit_label = ch_name + "_" + cluster_label_prefix + unit_type + str("%02i" % (unit_counter + 1))

            # Fill in table row-by-row
            row = SUMMARIZED_RESULTS.shape[0]
            SUMMARIZED_RESULTS.set_value(row, "name", unit_label)
            SUMMARIZED_RESULTS.set_value(row, "unit_type", unit_type)
            SUMMARIZED_RESULTS.set_value(row, "good", 1)
            SUMMARIZED_RESULTS.set_value(row, "peak_FWHM_ms", "%.3f" % FWHMp)
            SUMMARIZED_RESULTS.set_value(row, "trough_FWHM_ms", "%.3f" % FWHMt)
            SUMMARIZED_RESULTS.set_value(row, "peak_to_trough_ms", "%.3f" % P2T)
            SUMMARIZED_RESULTS.set_value(row, "peak_amplitude_mV", "%.3f" % peak_amplitude)
            SUMMARIZED_RESULTS.set_value(row, "trough_amplitude_mV", "%.3f" % trough_amplitude)
            SUMMARIZED_RESULTS.set_value(row, "FR", "%.1f" % FR)

    # Write file to disk
    SUMMARIZED_RESULTS.to_csv(fname, sep="\t", header=True, index=False, encoding="ascii", decimal=".")


def _compute_FWHM(waveform, x_peak):
    """

    :param waveform:
    :param x_peak:
    :return:
    """
    # 1. Find peak amplitude
    peak_amplitude = waveform[x_peak]

    # 2. Get half maximum
    HM = peak_amplitude / 2.

    # 3. Find points of intersection
    waveform0 = waveform - HM

    # 4a. Get the closest point before the peak
    point_before = x_peak - np.argmin(np.sign(waveform0[:x_peak][::-1]))

    # 4b. Get the closest point after the peak
    point_after = np.argmin(np.sign(waveform0[x_peak - 1:])) + x_peak - 1

    # 4c. Concatenate results
    idx_crossing = np.atleast_2d(np.array([point_before, point_after]))

    # 5. Get values at all these indices
    values_around_crossing = np.abs(waveform0[idx_crossing])

    # 6. Find minimum values
    rows = np.argmin(values_around_crossing, axis=0)

    # 7. Get the index of the crossing
    idx_crossing = [idx_crossing[rows[0], 0], idx_crossing[rows[1], 1]]

    # 8. Calculate full width
    FWHM = np.diff(idx_crossing)[0]

    return FWHM
