# -*- coding: utf-8 -*-
"""
This file contains the the class OSSA used to launch the GUI windows. Data is
loaded here, and default values for each graphical element are set here.
"""
# Numerical packages
import numpy as np
import pandas as pd

# Local repository
from Code.general_configs import GC
from Code.IO_operations.SQL_database import SQL_db
from Code.IO_operations.spikes_hdf5 import read_hdf5_data, update_hdf5_data, update_hdf5_OSSA_results, reset_hdf5_OSSA_results, read_sorting_summary_table, get_columns_from_table_description
from Code.Workflows.Spike_sorting_workflow.OSSA import configs as OSSA_configs
from Code.Workflows.Spike_sorting_workflow.OSSA.GUI_ChannelSelector import OSSA_ChannelSelector_GUI


class OSSA(object):
    """Open Spike Sorting Algorithm"""
    def __init__(self, launcher_GUI, animal_ID):
        # Reference the main GUI at the root level
        """Main class to load other GUIs of the OSSA suite.
        """
        self.launcher_GUI = launcher_GUI
        self.animal_ID = animal_ID
        # Hold in memory the exit flag
        self.quit_workflow = False
        self.exit_code = 0

        # Read sessions from database
        METADATA_sessions = SQL_db.read_table_where('sessions', None, animal_ID, 'animal_ID')
        self.recording_dates = METADATA_sessions['date'].values.copy()
        self.n_sessions = len(self.recording_dates)

        # Load default values for analysis and GUIs
        self.analysis = OSSA_configs.analysis
        self.plotted_features = OSSA_configs.plotted_features
        self.default = OSSA_configs.default
        self.keyboard_shortcuts = OSSA_configs.keyboard_shortcuts

        # Initialize variables
        self.ChannelSelector = None
        self.currently_showing = ''
        self.spike_detection_threshold = GC.spike_detection_threshold_MADs
        self.hdf5_columns = list(get_columns_from_table_description('SpikeData').keys())

        # Run the QtApp
        self.run()


    ############################################################################
    # Run GUI
    ############################################################################
    def run(self):
        """Launch the ChannelSelector GUI and start Qt event loop"""
        # Instantiate the ChannelSelector GUI
        self.ChannelSelector = OSSA_ChannelSelector_GUI(self)
        # Show the ChannelSelector GUI
        self.ChannelSelector.show_GUI()
        self.ChannelSelector.show_in_console_and_statusbar('Ready')


    ############################################################################
    # Interact with .hdf5 file
    ############################################################################
    def reset_OSSA_results(self):
        reset_hdf5_OSSA_results(self.spikes_hdf5_filename)
        # Update data in memory by reading the table once again from disk
        self.read_OSSA_results()

    def reset_cluster_ids_and_flags(self):
        raise Exception('Fix me!')
        for ch_name in self.signal_channel_name:
            # Read current data for this channel
            current_data = read_hdf5_data(self.spikes_hdf5_filename, ch_name, column_list=['cluster_id_auto', 'good_spike'], return_as_pandas=True)
            # Reset cluster id to the automatic output
            update_hdf5_data(self.spikes_hdf5_filename, ch_name, 'cluster_id_manual', current_data['cluster_id_auto'].values)
            # Reset the flag good spike to True
            current_data['good_spike'] = True
            update_hdf5_data(self.spikes_hdf5_filename, ch_name, 'good_spike', current_data['good_spike'].values)

    def read_OSSA_sorting_summary_table(self, session):
        return read_sorting_summary_table(self.spikes_hdf5_filename, session=session)
