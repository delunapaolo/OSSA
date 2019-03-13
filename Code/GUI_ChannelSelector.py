"""
First window to be opened when the OSSA GUI is launched. This GUI shows the map
 of channels and runs other windows for data refinement and sorting. When other
 GUIs are run, it checks the exit code of that GUI to decide what to do.
Exit codes are:
-1: initial value set by the ChannelSelector
0: from any GUI where user did not save the current progress
1: from NoiseRemoval GUI
2: from SpikeSorter GUI
"""
# System
from functools import partial
import gc as memory_garbage_collector

# Graphics
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle

# Numerical computation
import numpy as np

# Workflow
from Code.general_configs import GC
from Code.IO_operations.log4py import LOGGER
from Code.GUIs.GUI_configs import default as GUI_default
from Code.GUIs.GUI_utilities import initialize_field
from Code.IO_operations.spikes_hdf5 import read_hdf5_OSSA_results, update_hdf5_OSSA_results
from Code.IO_operations.get_filename_of import get_filename_of
from Code.IO_operations.files_and_folders import read_info_file, get_shank_of_intracortical_neuronal_channels, get_intracortical_neuronal_shanks
from Code.Workflows.Spike_sorting_workflow.OSSA.analysis_functions import compute_CCG_all_units
from Code.Workflows.Spike_sorting_workflow.OSSA.visualization import OSSA_window, OSSAHelpDialog, set_figure_style
from Code.Workflows.Spike_sorting_workflow.OSSA.GUI_NoiseRemoval import OSSA_NoiseRemoval_GUI
from Code.Workflows.Spike_sorting_workflow.OSSA.GUI_SpikeSorter import OSSA_SpikeSorter_GUI
from Code.third_party.circlify import circlify


class OSSA_ChannelSelector_GUI(object):
    def __init__(self, OSSA):
        """Receive a OSSA instance as input. That instance is created by running
        the OSSA_launcher."""
        LOGGER.info('OSSA: ChannelSelector GUI', decorate=True)

        # Reference OSSA instance at the root level. Now data and default values
        # set by the OSSA class will be available at self.OSSA
        self.OSSA = OSSA

        # Open window
        self.window = OSSA_window(title=self.OSSA.animal_ID)

        # Make the helper message for keyboard shortcuts
        shortcuts = """
        F1\tHelp
        %s\tReload from disk
        %s\tQuit OSSA and load next recording
        %s\tQuit OSSA and stop workflow
        
        Ctrl+A\tSelect all channels
        
        After highlighting a channel...
        Esc\tReset channel selection
        """ % (self.OSSA.keyboard_shortcuts['reload'],
               self.OSSA.keyboard_shortcuts['discard_and_close'],
               self.OSSA.keyboard_shortcuts['ChannelSelector']['quit_workflow'])
        self.helperWindow = OSSAHelpDialog(shortcuts)

        # Read waveform duration from general_configs
        spike_waveform_duration_ms = GC.spike_waveform_duration
        # Get intervals before / after peak
        self.waveform_before_peak_ms = np.abs(spike_waveform_duration_ms[0])
        self.waveform_after_peak_ms = np.abs(spike_waveform_duration_ms[1])
        self.waveform_before_peak = None
        self.waveform_after_peak = None
        self.spike_waveform_duration = None
        # Initialize fields
        self.current_session = self.OSSA.n_sessions -1
        self.current_session_date = self.OSSA.recording_dates[self.current_session]
        self.INFO = None
        self.n_channels_electrode = None
        self.n_channels_all_shanks = None
        self.channels_in_shank = None
        self.intracortical_neural_shanks = None
        self.n_shanks = 1
        self.spikes_hdf5_filename = None
        self.OSSA_results = None
        self.shanks_with_spikes = None
        self.fig = None
        self.ax = None
        self.lines = None
        self.channel_map = None
        self.colors = None
        self.buttons = None
        self.menu = None
        self.callbacks = None
        self.keyboard = None
        self.NoiseRemoval = None
        self.SpikeSorter = None
        self.shank_selection = None
        self.shank_selection_index = None
        self.shank_selection_names = None
        # Load data of last session
        self.load_data_of_current_session()
        # Get shanks suitable for spike sorting (i.e., with spikes)
        self._get_shanks_suitable_for_sorting()
        # Pre-allocate fields for graphical elements
        initialize_field(self, 'fig')
        initialize_field(self, 'ax')
        initialize_field(self, 'lines')
        initialize_field(self, 'colors')
        initialize_field(self, 'buttons')
        initialize_field(self, 'menu')
        initialize_field(self, 'callbacks')
        initialize_field(self, 'keyboard')
        # Unpack colors
        self.colors.active = colors.rgb2hex(self.OSSA.default['ChannelSelector']['color_channel_active'])
        self.colors.inactive = colors.rgb2hex(self.OSSA.default['ChannelSelector']['color_channel_inactive'])
        self.colors.highlight = colors.rgb2hex(self.OSSA.default['ChannelSelector']['color_channel_highlight'])
        self.colors.discard = colors.rgb2hex(self.OSSA.default['ChannelSelector']['color_channel_discard'])
        self.colors.keep = colors.rgb2hex(self.OSSA.default['ChannelSelector']['color_channel_keep'])
        self.colors.sorted = colors.rgb2hex(self.OSSA.default['ChannelSelector']['color_channel_sorted'])

        # Create main frame and draw channel map on it
        self._GUI_create_main_frame()
        self._draw_channel_map(initialize=True)
        # Reset the exit code and the variables that gets transferred between GUIs
        self._init_GUI_data()
        self.reset_GUI_data()
        # Destroy references to matplotlib's figures. These are now referenced
        # by the Qt event loop and need not to stay opened in matplotlib.
        [plt.close(self.fig.channels[sh][0]) for sh in range(self.n_shanks)]

        # Hide launcher GUI
        self.OSSA.launcher_GUI.hide()

        # This GUI does not show itself, but it's initialized by OSSA and then
        # made visible by other GUIs before deleting themselves.

        self.buttons.run_CCG_selected_channels.setEnabled(False)
        self.menu.session.setCurrentIndex(0)
        # self.shank_selection = np.zeros((7, ), dtype=bool)
        # self.shank_selection[4] = True
        # self.callback_run_NoiseRemoval()
        # self.callback_run_SpikeSorter()

    def load_data_of_current_session(self):
        info_filename = get_filename_of('info', self.OSSA.animal_ID, self.current_session_date)
        self.INFO = read_info_file(info_filename)
        # Keep subset of info that are relevant to spike sorting
        self.intracortical_neural_shanks = get_intracortical_neuronal_shanks(self.INFO)
        self.INFO['shank_names'] = np.array(self.INFO['shank_names'])[self.intracortical_neural_shanks]
        self.INFO['electrode_type'] = np.array(self.INFO['electrode_type'])[self.intracortical_neural_shanks]
        self.INFO['channel_index'] = np.array(self.INFO['channel_index'])[self.intracortical_neural_shanks]
        self.INFO['location'] = np.array(self.INFO['location'])[self.intracortical_neural_shanks]
        self.n_channels_electrode = np.array([len(ii) for ii in self.INFO['channel_index']], dtype=int)
        self.n_channels_all_shanks = int(np.sum(self.n_channels_electrode))
        self.channels_in_shank = get_shank_of_intracortical_neuronal_channels(self.INFO)
        self.n_shanks = len(self.intracortical_neural_shanks)
        # Get filenames of spike waveforms
        self.spikes_hdf5_filename = get_filename_of('spikes', self.OSSA.animal_ID, self.current_session_date, prefer_local=True)
        # Get previous results of spike sorting
        self._get_shanks_suitable_for_sorting()

        # Get intervals before / after peak
        self.waveform_before_peak = int(self.waveform_before_peak_ms / 1000. * self.INFO['sampling_frequency'])
        self.waveform_after_peak = int(self.waveform_after_peak_ms / 1000. * self.INFO['sampling_frequency'])
        # Waveform duration in samples
        self.spike_waveform_duration = (self.waveform_before_peak + self.waveform_after_peak + 1)

        # Get arrangement of shanks and electrodes
        channel_circles = list()
        for ii in range(self.n_shanks):
            if self.INFO['electrode_type'][ii] == 'tetrode':
                n_channels_in_shank = 4
            else:
                n_channels_in_shank = 1
            channel_circles.append(circlify(np.ones((n_channels_in_shank,)).tolist(), with_enclosure=True))
        self.channel_map = channel_circles


    def _GUI_create_main_frame(self):
        """Populates the main window with figures and buttons."""
        # Create empty frame
        self.qFrame = QtWidgets.QWidget(parent=self.window)
        # Make the background white
        p = self.qFrame.palette()
        p.setColor(self.qFrame.backgroundRole(), QtCore.Qt.white)
        self.qFrame.setPalette(p)
        self.qFrame.setAutoFillBackground(True)

        # Make vertical box to contain the shanks
        self.channel_map_widget = QtWidgets.QVBoxLayout()

        # Make drop-down menu for session selection
        session_label = QtWidgets.QLabel('Session')
        session_label.setFont(self.OSSA.default['ChannelSelector']['font'])
        self.menu.session = QtWidgets.QComboBox()
        self.menu.session.setFont(GUI_default['font_buttons'])
        [self.menu.session.addItem(i) for i in list(self.OSSA.recording_dates)]
        self.menu.session.setCurrentIndex(self.current_session)

        # Make buttons
        self.buttons.run_NoiseRemovalGUI = QtWidgets.QPushButton('Remove\nnoisy\nevents')
        self.buttons.run_NoiseRemovalGUI.setFont(self.OSSA.default['ChannelSelector']['font'])
        self.buttons.run_NoiseRemovalGUI.setToolTip('Launch the NoiseRemovalGUI')
        self.buttons.run_SpikeSorterGUI = QtWidgets.QPushButton('Sort\nspikes')
        self.buttons.run_SpikeSorterGUI.setFont(self.OSSA.default['ChannelSelector']['font'])
        self.buttons.run_SpikeSorterGUI.setToolTip('Launch the SpikeSorterGUI')
        self.buttons.run_CCG_selected_channels = QtWidgets.QPushButton('CCG\nselected\nchannels')
        self.buttons.run_CCG_selected_channels.setFont(self.OSSA.default['ChannelSelector']['font'])
        self.buttons.run_CCG_selected_channels.setToolTip('Compute the Cross-Correlogram\nof all spike trains\nin the selected channels.')
        # Make menubar
        self.menu.close_window = QtWidgets.QAction('Quit OSSA and load next', self.window)
        self.menu.close_window.setShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['discard_and_close']))
        self.menu.reload = QtWidgets.QAction('Reload from disk', self.window)
        self.menu.reload.setShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['reload']))
        self.menu.quit_workflow = QtWidgets.QAction('Quit OSSA and stop workflow', self.window)
        self.menu.quit_workflow.setShortcut(QtGui.QKeySequence('Ctrl+Q'))
        self.menu.menubar = self.window.menuBar()
        self.menu.file = self.menu.menubar.addMenu('File')
        self.menu.file.addAction(self.menu.reload)
        self.menu.file.addSeparator()
        self.menu.file.addAction(self.menu.close_window)
        self.menu.file.addSeparator()
        self.menu.file.addAction(self.menu.quit_workflow)
        # Make help menu
        self.menu.help = QtWidgets.QAction('Help', self.window)
        self.menu.help.setShortcut('F1')
        self.menu.help_menu = self.menu.menubar.addMenu('Help')
        self.menu.help_menu.addAction(self.menu.help)
        # Statusbar
        self.menu.statusBar = self.window.statusBar()
        self.menu.statusBar.setSizeGripEnabled(False)

        # Create a vertical widget layout where buttons are located
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(session_label)
        vbox.addWidget(self.menu.session)
        vbox.addStretch()
        vbox.addWidget(self.buttons.run_NoiseRemovalGUI)
        vbox.addWidget(self.buttons.run_SpikeSorterGUI)
        vbox.addStretch()
        vbox.addWidget(self.buttons.run_CCG_selected_channels)
        vbox.addStretch()
        # Create a horizontal widget layout to place plot on the left and buttons on the right
        hbox = QtWidgets.QHBoxLayout()
        hbox.addLayout(self.channel_map_widget)
        hbox.setStretch(0, 9)
        hbox.addLayout(vbox)
        hbox.setStretch(1, 1)
        # Assign layout and focus on it
        self.qFrame.setLayout(hbox)
        self.qFrame.setFocus()
        self.window.setCentralWidget(self.qFrame)

        # Make keyboard shortcut
        self.keyboard.reset_shank_selection = QtWidgets.QShortcut(QtGui.QKeySequence('Esc'), self.window)
        self.keyboard.reset_shank_selection.setAutoRepeat(False)
        self.keyboard.select_all_channels = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+A'), self.window)
        self.keyboard.select_all_channels.setAutoRepeat(False)
        # Enable shortcuts
        self.keyboard.reset_shank_selection.activated.connect(self.callback_reset_shank_selection)
        self.keyboard.select_all_channels.activated.connect(self.callback_select_all_shanks)

        # Connect callbacks to buttons
        self.menu.session.currentIndexChanged.connect(self.callback_session_changed)
        self.buttons.run_NoiseRemovalGUI.clicked.connect(self.callback_run_NoiseRemoval)
        self.buttons.run_SpikeSorterGUI.clicked.connect(self.callback_run_SpikeSorter)
        self.buttons.run_CCG_selected_channels.clicked.connect(self.callback_run_CCG_selected_channels)
        self.menu.reload.triggered.connect(self.callback_reload)
        self.menu.close_window.triggered.connect(partial(self.callback_close_window, quit_workflow=False))
        self.menu.quit_workflow.triggered.connect(partial(self.callback_close_window, quit_workflow=True))
        self.menu.help.triggered.connect(self.callback_show_help)


    def _draw_channel_map(self, initialize=True):
        """Draw the channel map in the matplotlib figure (on the left), add
        channel labels, set active and inactive (absent) channels with different
        colors; create also a highlighter point to mark currently selected
        channels, whose indices will be passed to other GUIs."""

        # Delete previous instance of channel map
        if not initialize:
            # https://stackoverflow.com/a/9383780
            def clearLayout(layout):
                if layout is not None:
                    while layout.count():
                        item = layout.takeAt(0)
                        widget = item.widget()
                        if widget is not None:
                            widget.deleteLater()
                        else:
                            clearLayout(item.layout())
            clearLayout(self.vbox_shanks)

        # Initialize fields for handles
        initialize_field(self.fig, 'channels', value_type='list', shape=(self.n_shanks, ))
        initialize_field(self.ax, 'channels', value_type='list', shape=(self.n_shanks, ))

        self.vbox_shanks = QtWidgets.QVBoxLayout()
        # Open matplotlib figure and assign it to a FigureCanvas
        set_figure_style((1, 1, 1), 'white')
        # Create a horizontal layout for each shank
        for sh in range(self.n_shanks):
            hbox_shank = QtWidgets.QHBoxLayout()
            # In the horizontal layout, make a vertical layout containing a plot and a text widget
            # Make plot widget
            f = Figure(facecolor='None', figsize=(2, 2))
            fc = FigureCanvas(f)
            fc.setParent(self.qFrame)
            fc.setMinimumSize(80, 80)
            self.fig.channels[sh] = [f, fc]
            self.ax.channels[sh] = self.fig.channels[sh][0].add_subplot(111)
            hbox_shank.addWidget(self.fig.channels[sh][1])
            # Write info about this shank
            txt = '%s (%s)\nchannels %s\n%s' % (self.INFO['shank_names'][sh], self.INFO['electrode_type'][sh], str(self.INFO['channel_index'][sh]), self.INFO['location'][sh])
            label = QtWidgets.QLabel(txt)
            label.setFont(self.OSSA.default['ChannelSelector']['font'])
            hbox_shank.addWidget(label)
            # Set relative size of widgets
            hbox_shank.setStretch(0, 1)
            hbox_shank.setStretch(1, 4)
            # Add widget to main layout
            self.vbox_shanks.addLayout(hbox_shank)
        # Assign layout to widget
        self.channel_map_widget.addLayout(self.vbox_shanks)

        # Initialize variables
        self.lines.shanks = list()
        self.lines.channels = list()
        self.lines.highlighter = list()
        # Draw shanks
        for sh in range(self.n_shanks):
            channel_circles = list(self.channel_map[sh])  # This copies the list
            # Get colors and picker tolerance
            if self.shanks_suitable_for_sorting[sh]:
                picker_tolerance = self.OSSA.default['ChannelSelector']['channel_picker_tolerance']
                color = self.OSSA.default['ChannelSelector']['color_channel_active']
            else:
                picker_tolerance = None
                color = self.OSSA.default['ChannelSelector']['color_channel_inactive']
            # Plot the circle for the shank
            x, y, r = channel_circles.pop(-1)
            self.lines.shanks.append(self.ax.channels[sh].add_patch(Circle((x, y), r, linewidth=2, fill=False, edgecolor='k', picker=picker_tolerance, label=str('%i' % sh), zorder=3)))
            self.lines.highlighter.append(self.ax.channels[sh].add_patch(Circle((x, y), r, fill=True, facecolor=self.colors.inactive, edgecolor='None', zorder=1, visible=False)))
            # Plot the channels
            lines_channels = list()
            for ch in range(len(channel_circles)):
                x, y, r = channel_circles[ch]
                lines_channels.append(self.ax.channels[sh].add_patch(Circle((x, y), r, linewidth=.5, fill=True, facecolor=color, edgecolor='k', label=str('%i' % sh), zorder=2)))
            self.lines.channels.append(lines_channels)

            # Fix axes appearance
            self.ax.channels[sh].set_xlim(-1.1, 1.1)
            self.ax.channels[sh].set_ylim(-1.1, 1.1)
            self.ax.channels[sh].set_aspect('equal')
            self.ax.channels[sh].axis('off')
            self.fig.channels[sh][0].tight_layout()
            # Redraw canvas
            self.ax.channels[sh].figure.canvas.draw()

        # Reset highlighter properties
        self.reset_GUI_channel_highlighter()
        # Make array with colors
        self.reset_GUI_channel_map_colors()
        # Connect callback for picking electrode shank
        self.callbacks.select_shank_handles = [self.fig.channels[sh][1].mpl_connect('pick_event', partial(self.callback_select_shank, selected_shank=sh)) for sh in range(self.n_shanks)]
        # Shrink window to content
        self.window.resize(self.window.minimumSizeHint())


    def _init_GUI_data(self):
        """<shank_selection> holds a boolean array of currently selected / highlighted shanks"""
        self.shank_selection = np.zeros((self.n_shanks,), dtype=np.bool)


    ############################################################################
    # User interaction with GUI
    ############################################################################
    def show_GUI(self):
        """This method should be only called from other GUIs and not itself."""
        # Run only if not currently visible, to avoid multiple clicks on the
        # 'show' button
        if self.OSSA.currently_showing != 'ChannelSelector':
            # Change flag immediately so this condition becomes False
            self.OSSA.currently_showing = 'ChannelSelector'
            # Update colors on channel map according to exit state of other GUIs
            self.change_GUI_channel_map_colors()
            # Reset GUI data
            self.reset_GUI_data()
            # Show window
            self.window.show()
            self.window.raise_()

    def hide_GUI(self):
        """The ChannelSelector GUI is hidden before opening other GUIs."""
        # Close children windows left open by the user (CCG, ISI)
        open_figs = plt.get_fignums()
        for fig in open_figs:
            plt.close(fig)
        # Close helper window, if left open
        if self.helperWindow.isVisible():
            self.helperWindow.deleteLater()
        # Hide this window
        self.window.hide()

    def reset_GUI_data(self):
        """Initialize GUI data: reset the exit code to -1, and delete the data
        that get transferred between GUIs."""
        # Initialize the GUI exit code to -1. More details in the docstring at
        # the top of the file.
        self.OSSA.exit_code = -1
        # Remove variables that hold data in memory
        self.NoiseRemoval = None
        self.SpikeSorter = None
        # Initialize variables to move between GUIs
        self.initialize_variables_for_other_GUIs()
        # Redraw canvas
        [self.ax.channels[sh].figure.canvas.draw() for sh in range(self.n_shanks)]
        # Empty memory from unused / non-referenced resources
        memory_garbage_collector.collect()

    def reset_shank_selection(self):
        """Reset the array of selected channels in memory and from the figure."""
        # Reset channel selection
        self.shank_selection = np.zeros((self.n_shanks,), dtype=np.bool)
        self._adapt_button_state()
        # Accordingly reset the channel highlighter
        self.reset_GUI_channel_highlighter()
        [self.ax.channels[sh].figure.canvas.draw() for sh in range(self.n_shanks)]

    def reset_GUI_channel_map_colors(self):
        """Initialize channel colors on the map."""
        self.lines.channel_map_color = np.array([self.colors.inactive] * self.n_shanks, dtype=object)
        self.lines.channel_map_color[self.shanks_suitable_for_sorting] = [self.colors.active]
        # Mark shanks that have been previously sorted as 'good', and mark
        # shanks that do not contain good spikes
        for idx in range(self.n_shanks):
            table_rows = np.where(self.OSSA_results['shank'] == self.INFO['shank_names'][idx])[0]
            # Restore color from last run
            if self.OSSA_results.loc[table_rows, 'SpikeSorter'].values[0]:
                self.lines.channel_map_color[idx] = self.colors.sorted
            elif self.OSSA_results.loc[table_rows, 'NoiseRemoval'].values[0]:
                self.lines.channel_map_color[idx] = self.colors.keep
        # Update plot
        for sh in range(self.n_shanks):
            for ch in range(len(self.lines.channels[sh])):
                self.lines.channels[sh][ch].set_facecolor(self.lines.channel_map_color[sh])

    def reset_GUI_channel_highlighter(self):
        """Initialize properties of the channel highlighter."""
        [i.set_visible(False) for i in self.lines.highlighter]

    def initialize_variables_for_other_GUIs(self):
        """'Initialize variables to move between GUIs"""
        self._adapt_button_state()
        # This function does not allow channels where no spikes were detected to
        # be used for spike sorting.
        self._get_shanks_suitable_for_sorting()

    def change_GUI_channel_map_colors(self):
        """If the user did something, signal it in the channel map."""
        # From NoiseRemoval GUI
        if self.OSSA.exit_code == 1:
            for idx in range(self.NoiseRemoval.n_shanks_to_refine):
                # Get number of 'good' spikes
                n_spikes = self.NoiseRemoval.DATA['suggested_good_spikes'][idx].shape[0]
                shank_idx = self.shank_selection_index[idx]
                if n_spikes == 0:
                    self.lines.channel_map_color[shank_idx] = self.colors.discard
                    self.shank_selection[shank_idx] = False
                else:
                    self.lines.channel_map_color[idx] = self.colors.keep
                [self.lines.channels[shank_idx][ch].set_facecolor(self.lines.channel_map_color[shank_idx]) for ch in range(self.NoiseRemoval.n_channels_these_shanks[idx])]
            self._adapt_button_state()

        # From SpikeSorter GUI
        elif self.OSSA.exit_code == 2:
            self.lines.channel_map_color[self.shank_selection_index] = self.colors.sorted
            shank_idx = self.shank_selection_index
            [self.lines.channels[shank_idx][ch].set_facecolor(self.lines.channel_map_color[shank_idx]) for ch in range(self.SpikeSorter.n_channels_this_shank)]

        # Reset highlighter state
        self.reset_shank_selection()

        # Update figure
        [self.ax.channels[sh].figure.canvas.draw() for sh in range(self.n_shanks)]

    def _get_shanks_suitable_for_sorting(self):
        """Mark which channels contain at least one 'good' spike, as decided by
        the user, that are suitable for spike sorting."""
        self.OSSA_results, shanks_suitable_for_sorting = read_hdf5_OSSA_results(self.spikes_hdf5_filename)
        self.shanks_suitable_for_sorting = np.in1d(self.INFO['shank_names'], shanks_suitable_for_sorting)

    def toggle_interaction(self, to):
        """Toggle all buttons from user interaction."""
        self.window.setEnabled(to)

    def _adapt_button_state(self):
        """Change button state according to number of selected channels."""
        n_shanks_selected = np.sum(self.shank_selection)

        # Toggle buttons
        self.buttons.run_NoiseRemovalGUI.setEnabled(n_shanks_selected > 0)
        self.buttons.run_SpikeSorterGUI.setEnabled(n_shanks_selected == 1)

        # Cross-correlograms
        # if n_channels_selected > 0:
        #     # Do these channels contain spikes? If more than 1 do, enable button
        #     clicked_channels = np.where(self.shank_selection)[0]
        #     selected_channels_with_spikes = np.array([self.sessions_suitable_for_sorting[i] for i in clicked_channels], dtype=int)
        #     if np.sum(selected_channels_with_spikes) > 0:
        #        self.buttons.run_CCG_selected_channels.setEnabled(True)
        # else:
        #     self.buttons.run_CCG_selected_channels.setEnabled(False)


    ############################################################################
    # Callback functions
    ############################################################################
    def callback_session_changed(self):
        """This function is called whenever the user chooses a different session
        from the drop-down menu."""
        # Disable interaction
        self.toggle_interaction(to=False)
        # Re-load data
        self.current_session = self.menu.session.currentIndex()
        self.current_session_date = self.OSSA.recording_dates[self.current_session]
        self.load_data_of_current_session()
        self._draw_channel_map(initialize=False)
        # Reset the currently selected channels
        self.reset_shank_selection()
        self.show_in_console_and_statusbar('Ready')
        # Enable interaction
        self.toggle_interaction(to=True)

    def callback_select_shank(self, _, selected_shank=None):
        """This function is called every time a channel is clicked on the
        channel map. The event variable is created and passed by matplotlib."""
        # Get the clicked shank from clicked artist
        if not self.shanks_suitable_for_sorting[selected_shank]:
            return
        # Toggle highlighter
        new_state = not self.shank_selection[selected_shank]
        self.lines.highlighter[selected_shank].set_visible(new_state)
        self.shank_selection[selected_shank] = new_state

        # Update plot
        self.ax.channels[selected_shank].figure.canvas.draw_idle()
        self._adapt_button_state()

    def callback_reset_shank_selection(self):
        """Callback to deselect all channels. By default is triggered by the Esc key."""
        # Reset channel selection
        self.reset_shank_selection()

    def callback_select_all_shanks(self):
        """Callback to select all channels. By default is triggered by the Ctrl+A
        key combination."""
        # First, reset channel selection
        self.reset_shank_selection()
        # Select all spikes
        for sh in range(self.n_shanks):
            self.shank_selection[sh] = True
            self.lines.highlighter[sh].set_visible(True)
        [self.ax.channels[sh].figure.canvas.draw_idle() for sh in range(self.n_shanks)]
        self._adapt_button_state()

    def callback_run_NoiseRemoval(self):
        """This function is called every time the button 'Remove noisy spikes'
        is clicked."""
        # Update the list of channels that will need refinement
        self.shank_selection_index = np.where(self.shank_selection)[0]
        self.shank_selection_names = self.INFO['shank_names'][self.shank_selection_index]
        LOGGER.info('Started manual refinement for shanks %s' % self.shank_selection_names)
        # Hide itself
        self.hide_GUI()
        # Run other GUI
        self.NoiseRemoval = OSSA_NoiseRemoval_GUI(self)
        self.NoiseRemoval.show_GUI()

    def callback_run_SpikeSorter(self):
        """This function is called every time the button 'Sort spikes' is
        clicked."""
        # Get names and indices of spikes to sort
        self.shank_selection_index = np.where(self.shank_selection)[0][0]
        self.shank_selection_names = self.INFO['shank_names'][self.shank_selection_index]
        LOGGER.info('Started manual clustering for channel %s' % self.shank_selection_names)
        # Hide itself
        self.hide_GUI()
        # Run other GUI
        self.SpikeSorter = OSSA_SpikeSorter_GUI(self)
        self.SpikeSorter.show_GUI()

    def callback_run_CCG_selected_channels(self):
        """This function computes the cross-correlogram between spike trains
        belonging to different channels."""
        # Get names and indices of channels to analyze
        self.shank_selection_index = np.where(self.shank_selection)[0] + 1
        # Skip channels that have no spikes
        OSSA_results = read_hdf5_OSSA_results(self.OSSA.spikes_hdf5_filename)[0]
        channels_with_spikes = np.where(np.in1d(self.shank_selection_names, OSSA_results.loc[np.where(OSSA_results['contains_spikes'] == True)[0], 'channel'].values))[0]
        self.shank_selection_index = self.shank_selection_index[channels_with_spikes]
        self.shank_selection_names = list(self.shank_selection_names[channels_with_spikes])
        # Log
        self.show_in_console_and_statusbar('Computing CCGs between channels %s' % self.shank_selection_names)
        # Compute cross-correlograms between channels, and return figure handle
        fig = compute_CCG_all_units(self.OSSA.spikes_hdf5_filename, self.OSSA.info_filename, self.shank_selection_names, write_spyking_circus=False, read_from_text_file=False, reset_cluster_id=False, print_pdf=False)
        # Show figure and log outcome
        fig.show()
        self.show_in_console_and_statusbar('Ready')

    def callback_reload(self):
        """Callback for when the user wants to reload the data."""
        # Replace the data in memory with the output from SpikeDetectionWorkflow.
        # Principal components are not reset because the user can do so, if he
        # or she wishes.
        self.show_in_console_and_statusbar('Reloading data from disk')
        self.toggle_interaction(to=False)

        # Reset OSSA_results, cluster ids, 'good' flag, and GUI data
        self.OSSA.reset_OSSA_results()
        self.OSSA.reset_cluster_ids_and_flags()
        self.reset_GUI_data()

        # Enable interaction
        self.toggle_interaction(to=True)
        # Reset the currently selected channels
        self.reset_shank_selection()
        self.reset_GUI_channel_highlighter()
        self.reset_GUI_channel_map_colors()
        [self.ax.channels[sh].figure.canvas.draw() for sh in range(self.n_shanks)]
        self.show_in_console_and_statusbar('Ready')

    def callback_close_window(self, quit_workflow):
        """Callback for when the user closes the GUI."""
        LOGGER.info('Quitting OSSA')
        # Close window
        self.window.allowed_to_close = True
        self.window.close()
        # Pass the quit_workflow flag to OSSA object
        self.OSSA.quit_workflow = quit_workflow

    def callback_show_help(self):
        """Callback to make the helper window appear."""
        self.helperWindow.on_show()
        LOGGER.trace('Showing helper')


    ############################################################################
    # Helper functions
    ############################################################################
    def show_in_console_and_statusbar(self, msg):
        """Shortcut function to display a message in both statusbar and as displayed"""
        LOGGER.info(msg)
        self.menu.statusBar.showMessage(msg)


    ############################################################################
    # Interact with .hdf5 file
    ############################################################################
    def update_OSSA_results(self, shank_name, from_GUI, contains_spikes):
        update_hdf5_OSSA_results(self.spikes_hdf5_filename, shank_name, from_GUI, contains_spikes)
        # Update data in memory by reading the table once again from disk
        self._get_shanks_suitable_for_sorting()
