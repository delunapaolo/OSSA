# -*- coding: utf-8 -*-
"""
GUI for refinement of spike sorting.
"""

# System
from functools import partial
from copy import deepcopy as backup

# Graphics
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Numerical computation
import numpy as np

# Local repository
from Code.IO_operations.log4py import LOGGER
from Code.GUIs.GUI_configs import default_colors
from Code.GUIs.GUI_utilities import initialize_field
from Code.Workflows.Spike_sorting_workflow.OSSA.visualization import SelectFromCollection, OSSAFigureCanvas, OSSAHelpDialog, make_QSplitter, set_figure_style, translate_feature_to_axis_label, translate_feature_list_to_tooltip, OSSA_window
from Code.Workflows.Spike_sorting_workflow.OSSA.analysis_functions import compute_array_range, divide0
from Code.IO_operations.spikes_hdf5 import read_hdf5_data, update_hdf5_data, read_waveforms


class OSSA_NoiseRemoval_GUI(object):
    def __init__(self, ChannelSelector):
        LOGGER.info('OSSA: NoiseRemoval GUI')

        # Reference some objects at the root level.
        self.OSSA = ChannelSelector.OSSA
        self.CS = ChannelSelector

        # Put GUI input in a table with the original index of the channel in the
        # data and its ordinal index in the GUI
        self.n_shanks_to_refine = int(self.CS.shank_selection_index.shape[0])
        self.shank_selection_table = np.vstack((self.CS.shank_selection_index + 1, np.arange(self.n_shanks_to_refine))).transpose()
        self.channels_of_shank = None

        # Initialize fields that will contain other handles
        self.all_spike_indices = list()
        self.HISTORY = None
        self.DATA = None
        self.spike_polarity = self.CS.INFO['detection_threshold_polarity']
        self.fig = None
        self.matplotlib_canvas = None
        self.matplotlib_toolbar = None
        self.ax = None
        self.ax_lims = None
        self.lines = None
        self.last_selected_line = None
        self.text = None
        self.buttons = None
        self.callbacks = None
        self.menu = None
        self.keyboard = None
        # Callback variables
        self.currently_panning = False
        self.currently_selecting_spikes = False
        self.polygon_selectors = None
        self.waveform_plot_style = self.OSSA.default['NoiseRemoval']['waveform_plot_style']
        # Initialize all instance fields and subfields
        self._initialize_handles()

        # Is this the first run?
        self.use_suggestion = self._is_first_run()

        # Load data
        self._load_data_of_shanks()

        # Make the helper message for keyboard shortcuts
        shortcuts = """
        F1\tHelp
        %s\tSave and close
        %s\tDiscard and close
        %s\tReload last save
        %s / %s\tUndo / Redo (up to %i times)
        
        %s\tToggle suggestions
        %s\tKeep spikes
        %s\tDiscard spikes
        %s\tPan / zoom
        %s\tReset zoom
        
        While selecting spikes from clicked axes...
        Ctrl+A\tSelect all
        Enter\tAccept selection
        Esc\tRestart current selection
        Esc+Esc\tCancel all selections
        """ % (self.OSSA.keyboard_shortcuts['save_and_close'],
               self.OSSA.keyboard_shortcuts['discard_and_close'],
               self.OSSA.keyboard_shortcuts['reload'],
               self.OSSA.keyboard_shortcuts['undo'],
               self.OSSA.keyboard_shortcuts['redo'],
               self.OSSA.analysis['history_size'],
               self.OSSA.keyboard_shortcuts['NoiseRemoval']['toggle_suggestions'],
               self.OSSA.keyboard_shortcuts['NoiseRemoval']['keep_spikes'],
               self.OSSA.keyboard_shortcuts['NoiseRemoval']['discard_spikes'],
               self.OSSA.keyboard_shortcuts['pan_zoom'],
               self.OSSA.keyboard_shortcuts['reset_zoom'])

        # Make windows
        self.window = OSSA_window(title='%s - session %s' % (self.OSSA.animal_ID, self.OSSA.recording_dates[self.CS.current_session]))
        self.helperWindow = OSSAHelpDialog(shortcuts)

        # Create main frame and draw data on it
        self._GUI_create_main_frame()
        self.initialize_plots()
        self._cleanup_init()
        # Log end of __init__
        LOGGER.info('ok', append=True)

        # This GUI does not show itself, but it's initialized by the ChannelSelector
        # GUI and made visible from there.


    def _initialize_handles(self):
        """Create all the attributes that will hold graphical elements"""
        # These are the names of the plots. If a new one needs to be added, this
        # is the place
        self._plot_names = ['stability', 'feature', 'waveform']
        # Names of buttons
        self._button_names_plot = ['discard_all', 'keep_all', 'select_to_discard', 'select_to_keep', 'apply_selection', 'cancel_selection']
        # Data fields
        initialize_field(self, 'DATA')
        initialize_field(self.DATA, ['stability_x', 'stability_y', 'feature_x', 'feature_y', 'firing_rate', 'suggest_spikes_array', 'suggested_good_spikes', 'suggested_bad_spikes', 'time'], 'list', shape=self.n_shanks_to_refine)
        # History manager
        self.history_manager(action='initialize')
        # Graphic handles
        initialize_field(self, 'fig')
        initialize_field(self.fig, self._plot_names, 'list', shape=self.n_shanks_to_refine)
        initialize_field(self, 'matplotlib_canvas')
        initialize_field(self.matplotlib_canvas, self._plot_names, 'list', shape=self.n_shanks_to_refine)
        initialize_field(self, 'matplotlib_toolbar')
        initialize_field(self.matplotlib_toolbar, self._plot_names, 'list', shape=self.n_shanks_to_refine)
        initialize_field(self, 'ax')
        initialize_field(self.ax, self._plot_names, 'list', shape=self.n_shanks_to_refine)
        initialize_field(self, 'ax_lims')
        for plt_name in self._plot_names:
            self.ax_lims[plt_name] = np.empty((self.n_shanks_to_refine, 2), dtype=object)
        self.channels_these_shanks = [self.CS.INFO['channel_index'][self.CS.shank_selection_index[sh]] for sh in range(self.n_shanks_to_refine)]
        self.n_channels_these_shanks = [self.CS.n_channels_electrode[self.CS.shank_selection_index[sh]] for sh in range(self.n_shanks_to_refine)]
        self.waveform_plot_x = list()
        for sh in range(self.n_shanks_to_refine):
            x = np.arange(self.CS.spike_waveform_duration)
            if self.waveform_plot_style == 'horizontal':
                self.waveform_plot_x.append([x + (self.CS.spike_waveform_duration + 1) * ch for ch in range(self.n_channels_these_shanks[sh])])
        initialize_field(self, 'lines')
        initialize_field(self.lines, self._plot_names, 'list', shape=self.n_shanks_to_refine)
        initialize_field(self, 'last_selected_line')
        initialize_field(self.last_selected_line, self._plot_names[:2], 'list', shape=self.n_shanks_to_refine)
        initialize_field(self, 'text')
        initialize_field(self.text, self._plot_names[:2], 'list', shape=self.n_shanks_to_refine)
        initialize_field(self, 'buttons')
        initialize_field(self.buttons, self._button_names_plot, 'list', shape=self.n_shanks_to_refine)
        initialize_field(self, 'menu')
        initialize_field(self, 'keyboard')
        initialize_field(self, 'callbacks')
        initialize_field(self.callbacks, 'pick_spike')
        initialize_field(self.callbacks.pick_spike, self._plot_names[:2], 'list', shape=self.n_shanks_to_refine)
        initialize_field(self, 'polygon_selectors')
        initialize_field(self.polygon_selectors, ['stability', 'feature'], 'list', shape=0)

    def _load_data_of_shanks(self):
        """Load data of channels of interest. Also, suggest events that might not
        be good as spikes.
        """
        # Read .waveforms files
        self.WAVEFORMS = [read_waveforms(self.CS.spikes_hdf5_filename, shank_name) for shank_name in self.CS.shank_selection_names]
        # Get features to plot
        features_to_read = list(set(list(['time', 'timestamp', 'peak_amplitude' , 'peak_amplitude_SD', 'peak_to_trough_mV',
                                          self.OSSA.plotted_features['NoiseRemoval']['y']['stability'],
                                          self.OSSA.plotted_features['NoiseRemoval']['x']['feature'],
                                          self.OSSA.plotted_features['NoiseRemoval']['y']['feature']])))
        allowed_features = list(set(features_to_read).intersection(set(self.OSSA.hdf5_columns)))
        did_check_features_to_read = False
        # Get indices of all spikes
        for shank in range(self.n_shanks_to_refine):
            # Get spike features
            data = read_hdf5_data(self.CS.spikes_hdf5_filename, self.CS.shank_selection_names[shank], column_list=allowed_features, return_as_pandas=True)
            data['trough_amplitude'] = (data['peak_to_trough_mV'] - np.abs(data['peak_amplitude'])) * -np.sign(data['peak_amplitude'])
            # Get number of spikes
            n_spikes = data.shape[0]

            # Get variability of baseline
            data['variability_baseline'] = divide0(data['peak_amplitude'], data['peak_amplitude_SD'], 0)
            # Calculate the trough amplitude in SD units
            data['trough_amplitude_SD'] = divide0(data['trough_amplitude'], data['variability_baseline'], 0)

            # Before proceeding, make sure that we have all the features
            if not did_check_features_to_read:
                did_check_features_to_read = True
                features_in_memory = list(data.columns)
                if not all([f in features_in_memory for f in features_to_read]):
                    raise Exception('ERROR: Some features cannot be found. Please check OSSA_configs')

            # Create a suggestion for good spikes
            # If this is the first run, suggestions will be turned on. In that
            # case, suggest as 'bad spike' based on some criteria
            if self.use_suggestion:
                self.DATA['suggest_spikes_array'][shank] = np.ones((n_spikes, ), dtype=bool)
                # Remove spikes whose amplitude is lower than the threshold
                self.DATA['suggest_spikes_array'][shank][np.abs(data['peak_amplitude_SD']) < self.OSSA.spike_detection_threshold] = 0
                # Remove spikes whose trough is not opposite of their peak in terms
                # of polarity, i.e., the waveform 'goes up and down'
                self.DATA['suggest_spikes_array'][shank][np.sign(data['peak_amplitude_SD']) == np.sign(data['trough_amplitude_SD'])] = 0

            else:  # otherwise, read data from disk
                self.DATA['suggest_spikes_array'][shank] = read_hdf5_data(self.CS.spikes_hdf5_filename, self.CS.shank_selection_names[shank], column_list='good_spike')

            # Keep data in memory
            self.DATA['time'][shank] = data['time'].values
            self.DATA['stability_x'][shank] = data['timestamp'].values
            self.DATA['stability_y'][shank] = data[self.OSSA.plotted_features['NoiseRemoval']['y']['stability']].values
            self.DATA['feature_x'][shank] = data[self.OSSA.plotted_features['NoiseRemoval']['x']['feature']].values
            self.DATA['feature_y'][shank] = data[self.OSSA.plotted_features['NoiseRemoval']['y']['feature']].values

        # Keep in memory the initial state of the data
        self.HISTORY.first_state = backup(self.DATA['suggest_spikes_array'])
        self.HISTORY.current_state = backup(self.DATA['suggest_spikes_array'])

        # Get firing rate of good spikes
        self.update_suggested_spike_indices()
        self.update_firing_rates()

    def update_suggested_spike_indices(self):
        for ch in range(self.n_shanks_to_refine):
            self.DATA['suggested_good_spikes'][ch] = np.where(self.DATA['suggest_spikes_array'][ch] == True)[0]
            self.DATA['suggested_bad_spikes'][ch] = np.where(self.DATA['suggest_spikes_array'][ch] == False)[0]

    def update_firing_rates(self):
        """Calculate the firing rate of a spike train, in Hz."""
        for sh in range(self.n_shanks_to_refine):
            n_spikes = self.DATA['suggest_spikes_array'][sh].shape[0]
            n_good = self.DATA['suggested_good_spikes'][sh].shape[0]
            n_bad = n_spikes - n_good
            firing_rate_good = n_good / self.CS.INFO['recording_duration']
            firing_rate_bad = n_bad / self.CS.INFO['recording_duration']
            self.DATA['firing_rate'][sh] = [firing_rate_good, firing_rate_bad]

    def _cleanup_init(self):
        """Routines to be performed at the end of __init__ function."""
        # Destroy references to matplotlib's figures. These are now referenced
        # by the Qt event loop and don't need to be opened by the matplotlib's
        # renderer.
        for plt_name in self._plot_names:
            for fig in self.fig[plt_name]:
                plt.close(fig)

    def _is_first_run(self):
        """Determine whether this is the first time user runs spike sorting on
        these channels."""
        rows = np.where(np.in1d(self.CS.OSSA_results['shank'], self.CS.shank_selection_names))[0]
        all_processed = self.CS.OSSA_results.loc[rows, 'NoiseRemoval'].all()
        return not all_processed

    def _GUI_create_main_frame(self):
        # Create empty widget
        self.qFrame = QtWidgets.QWidget(parent=self.window)
        # Make the background white
        p = self.qFrame.palette()
        p.setColor(self.qFrame.backgroundRole(), QtCore.Qt.white)
        self.qFrame.setPalette(p)
        self.qFrame.setAutoFillBackground(True)

        # Open matplotlib figures and assign them to FigureCanvas objects
        set_figure_style(self.OSSA.default['axes_color'])
        for sh in range(self.n_shanks_to_refine):
            # Stability plot
            self.fig.stability[sh] = Figure(facecolor='w')
            self.matplotlib_canvas.stability[sh] = OSSAFigureCanvas(self.fig.stability[sh], self.qFrame)
            self.matplotlib_toolbar.stability[sh] = NavigationToolbar(self.matplotlib_canvas.stability[sh], self.qFrame)
            self.matplotlib_toolbar.stability[sh].hide()
            self.ax.stability[sh] = self.fig.stability[sh].add_subplot(1, 1, 1)

            # Feature plot
            self.fig.feature[sh] = Figure(facecolor='w')
            self.matplotlib_canvas.feature[sh] = OSSAFigureCanvas(self.fig.feature[sh], self.qFrame)
            self.matplotlib_toolbar.feature[sh] = NavigationToolbar(self.matplotlib_canvas.feature[sh], self.qFrame)
            self.matplotlib_toolbar.feature[sh].hide()
            self.ax.feature[sh] = self.fig.feature[sh].add_subplot(1, 1, 1)

            # Waveform plot
            self.fig.waveform[sh] = Figure(facecolor='w')
            self.matplotlib_canvas.waveform[sh] = OSSAFigureCanvas(self.fig.waveform[sh], self.qFrame)
            self.matplotlib_toolbar.waveform[sh] = NavigationToolbar(self.matplotlib_canvas.waveform[sh], self.qFrame)
            self.matplotlib_toolbar.waveform[sh].hide()
            # Make a reference to the first axis channel to share x-limits with
            # all the other channels
            if sh == 0:
                waveform_plot_share_ax = None
            else:
                waveform_plot_share_ax = self.ax.waveform[0]
            # Axis space
            self.ax.waveform[sh] = self.fig.waveform[sh].add_subplot(1, 1, 1, sharex=waveform_plot_share_ax, sharey=waveform_plot_share_ax)

        # Change resize policy of the graphs to limit their minimum size during resizing
        for plt_name in self._plot_names:
            for canvas in self.matplotlib_canvas[plt_name]:
                canvas.setMinimumSize(self.OSSA.default['widget_minimum_size'], self.OSSA.default['widget_minimum_size'])
                canvas.updateGeometry()

        # Make a 'splitter' layout
        splitter_layout = make_QSplitter('hor')
        layout_columns = np.empty(len(self._plot_names, ), dtype=object)
        for plt_col, plt_name in enumerate(self._plot_names):
            layout_columns[plt_col] = QtWidgets.QVBoxLayout()
            for ch in range(self.n_shanks_to_refine):
                layout_columns[plt_col].addWidget(self.matplotlib_canvas[plt_name][ch])
            # Enclose the column in a plain widget
            column_widget = QtWidgets.QWidget()
            column_widget.setLayout(layout_columns[plt_col])
            # Assign the widget to the splitter
            splitter_layout.addWidget(column_widget)

        # Set the initial size
        splitter_layout.setStretchFactor(0, 7)
        splitter_layout.setStretchFactor(1, 2)
        splitter_layout.setStretchFactor(2, 2)

        # Make menubar's actions
        self.menu.cancel = QtWidgets.QAction('Discard and close window', self.window)
        self.menu.cancel.setShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['discard_and_close']))
        self.menu.save = QtWidgets.QAction('Save and close window', self.window)
        self.menu.save.setShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['save_and_close']))
        self.menu.reload = QtWidgets.QAction('Reload last save', self.window)
        self.menu.reload.setShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['reload']))
        self.menu.undo = QtWidgets.QAction('Undo', self.window)
        self.menu.undo.setShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['undo']))
        self.menu.redo = QtWidgets.QAction('Redo', self.window)
        self.menu.redo.setShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['redo']))
        # Make help menu
        self.menu.help = QtWidgets.QAction('Help', self.window)
        self.menu.help.setShortcut('F1')
        # Make toolbar's actions
        self.menu.toolbar_reset_zoom = QtWidgets.QToolButton(self.window)
        self.menu.toolbar_reset_zoom.setIcon(QtGui.QIcon(QtGui.QPixmap(self.OSSA.default['icon_home'])))
        self.menu.toolbar_reset_zoom.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.menu.toolbar_reset_zoom.setCheckable(False)
        self.menu.toolbar_reset_zoom.setToolTip('Reset zoom in all plots')
        self.menu.toolbar_pan_zoom = QtWidgets.QToolButton(self.window)
        self.menu.toolbar_pan_zoom.setIcon(QtGui.QIcon(QtGui.QPixmap(self.OSSA.default['icon_magnifying_glass'])))
        self.menu.toolbar_pan_zoom.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.menu.toolbar_pan_zoom.setCheckable(True)
        self.menu.toolbar_pan_zoom.setToolTip('Toggle pan/zoom tool in all plots')
        self.menu.toolbar_suggest = QtWidgets.QToolButton(self.window)
        self.menu.toolbar_suggest.setIcon(QtGui.QIcon(QtGui.QPixmap(self.OSSA.default['icon_visible'])))
        self.menu.toolbar_suggest.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.menu.toolbar_suggest.setCheckable(True)
        self.menu.toolbar_suggest.setToolTip('Toggle spike suggestions')
        # Toolbar's actions to interact with data
        self.menu.select_to_discard = QtWidgets.QToolButton(self.window)
        self.menu.select_to_discard.setIcon(QtGui.QIcon(QtGui.QPixmap(self.OSSA.default['icon_polygon'])))
        self.menu.select_to_discard.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(self.OSSA.default['NoiseRemoval']['channel_button_color_discard'], dtype=float)*255))))
        self.menu.select_to_discard.setToolTip('Enable selector of spikes to discard')
        self.menu.select_to_keep = QtWidgets.QToolButton(self.window)
        self.menu.select_to_keep.setIcon(QtGui.QIcon(QtGui.QPixmap(self.OSSA.default['icon_polygon'])))
        self.menu.select_to_keep.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(self.OSSA.default['NoiseRemoval']['channel_button_color_keep'], dtype=float)*255))))
        self.menu.select_to_keep.setToolTip('Enable selector of spikes to keep')

        # Make menubar and statusbar
        self.menu.menubar = self.window.menuBar()
        self.menu.file = self.menu.menubar.addMenu('File')
        self.menu.file.addAction(self.menu.save)
        self.menu.file.addSeparator()
        self.menu.file.addAction(self.menu.cancel)
        self.menu.file.addSeparator()
        self.menu.file.addAction(self.menu.reload)
        self.menu.edit = self.menu.menubar.addMenu('Edit')
        self.menu.edit.addAction(self.menu.undo)
        self.menu.file.addSeparator()
        self.menu.edit.addAction(self.menu.redo)
        self.menu.help_menu = self.menu.menubar.addMenu('Help')
        self.menu.help_menu.addAction(self.menu.help)
        # Statusbar
        self.menu.statusBar = self.window.statusBar()
        self.menu.statusBar.setSizeGripEnabled(False)
        # Toolbar
        self.menu.toolBar = QtWidgets.QToolBar('Tools')
        self.menu.toolBar.setStyleSheet('QToolBar {background: white}')
        self.window.addToolBar(QtCore.Qt.LeftToolBarArea, self.menu.toolBar)
        self.menu.toolBar.setFloatable(False)
        self.menu.toolBar.setMovable(True)
        self.menu.toolBar.addWidget(self.menu.toolbar_reset_zoom)
        self.menu.toolBar.addWidget(self.menu.toolbar_pan_zoom)
        self.menu.toolBar.addSeparator()
        self.menu.toolBar.addWidget(self.menu.toolbar_suggest)
        self.menu.toolBar.addSeparator()
        self.menu.toolBar.addWidget(self.menu.select_to_discard)
        self.menu.toolBar.addWidget(self.menu.select_to_keep)

        # Set state
        self.toggle_suggest_button(self.use_suggestion)
        self.toggle_button_pan_zoom(False)

        # Put the splitter and the window buttons in a vertical layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(splitter_layout)
        # Assign layout to frame and focus on it
        self.qFrame.setLayout(layout)
        self.qFrame.setFocus()
        self.window.setCentralWidget(self.qFrame)

        # Connect callbacks
        # GUI resizing
        self.window.resized.connect(self.callback_on_resize)
        # Point pickers
        for idx in range(self.n_shanks_to_refine):
            self.callbacks.pick_spike.stability[idx] = self.fig.stability[idx].canvas.mpl_connect('pick_event', partial(self.callback_show_clicked_waveform, from_plot='stability'))
            self.callbacks.pick_spike.feature[idx] = self.fig.feature[idx].canvas.mpl_connect('pick_event', partial(self.callback_show_clicked_waveform, from_plot='feature'))
        # Menubar
        self.menu.help.triggered.connect(self.callback_show_help)
        self.menu.cancel.triggered.connect(partial(self.callback_quit, save=False))
        self.menu.save.triggered.connect(partial(self.callback_quit, save=True))
        self.menu.reload.triggered.connect(self.callback_reload)
        self.menu.undo.triggered.connect(partial(self.history_manager, action='undo'))
        self.menu.redo.triggered.connect(partial(self.history_manager, action='redo'))
        # Toolbar
        self.menu.toolbar_pan_zoom.clicked.connect(self.callback_pan_zoom_figure)
        self.menu.toolbar_reset_zoom.clicked.connect(self.callback_reset_zoom)
        self.menu.toolbar_suggest.clicked.connect(self.callback_toggle_suggestions)
        self.menu.select_to_discard.clicked.connect(self.callback_select_to_discard)
        self.menu.select_to_keep.clicked.connect(self.callback_select_to_keep)
        # Make keyboard shortcuts
        self.keyboard.keep_spikes = QtWidgets.QShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['NoiseRemoval']['keep_spikes']), self.window)
        self.keyboard.keep_spikes.setAutoRepeat(False)
        self.keyboard.discard_spikes = QtWidgets.QShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['NoiseRemoval']['discard_spikes']), self.window)
        self.keyboard.discard_spikes.setAutoRepeat(False)
        self.keyboard.pan_zoom_figure = QtWidgets.QShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['pan_zoom']), self.window)
        self.keyboard.pan_zoom_figure.setAutoRepeat(False)
        self.keyboard.reset_zoom = QtWidgets.QShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['reset_zoom']), self.window)
        self.keyboard.reset_zoom.setAutoRepeat(False)
        self.keyboard.toggle_suggestions = QtWidgets.QShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['NoiseRemoval']['toggle_suggestions']), self.window)
        self.keyboard.toggle_suggestions.setAutoRepeat(False)
        # Enable shortcuts
        self.keyboard.keep_spikes.activated.connect(partial(self.callback_select_to_keep, from_keyboard=True))
        self.keyboard.discard_spikes.activated.connect(partial(self.callback_select_to_discard, from_keyboard=True))
        self.keyboard.pan_zoom_figure.activated.connect(self.callback_pan_zoom_figure)
        self.keyboard.reset_zoom.activated.connect(self.callback_reset_zoom)
        self.keyboard.toggle_suggestions.activated.connect(self.callback_toggle_suggestions)

    def initialize_plots(self):
        """Initialize graphical handles for data"""
        for idx in range(self.n_shanks_to_refine):
            # Get the index of the channel and of the axes
            sh_idx = self.shank_selection_table[idx, 0]
            ax_idx = self.shank_selection_table[idx, 1]
            # Make a 'plot label' to store axes and channel indices
            label_channel = '%i_%i_' % (sh_idx, ax_idx)

            # Stability plot
            self.lines.stability[ax_idx] = [self.ax.stability[ax_idx].plot(0, 0, alpha=1., marker='.', markersize=self.OSSA.default['NoiseRemoval']['size_bad_spikes'], linestyle='None', markeredgecolor='None', markerfacecolor=self.OSSA.default['NoiseRemoval']['color_bad_spikes'], markeredgewidth=2, picker=5, zorder=1, label=label_channel + '0')[0],
                                            self.ax.stability[ax_idx].plot(0, 0, alpha=1., marker='.', markersize=self.OSSA.default['scatter_size'], linestyle='None', markeredgecolor='None', markerfacecolor=self.OSSA.default['NoiseRemoval']['color_good_spikes'], markeredgewidth=2, picker=5, zorder=2, label=label_channel + '1')[0]]
            # Add a visible marker for the selected point
            self.last_selected_line.stability[ax_idx] = self.ax.stability[ax_idx].plot(-1000, 0, linestyle='None', marker='o', markersize=self.OSSA.default['last_selected_point_size'], markerfacecolor='r', markeredgecolor='w', markeredgewidth=1.5, zorder=3, scalex=False, scaley=False)[0]
            # Plot threshold(s)
            if self.OSSA.plotted_features['NoiseRemoval']['y']['stability'] == 'peak_amplitude_SD':
                if self.spike_polarity[ax_idx] == 'positive' or self.spike_polarity[ax_idx] == 'bipolar':
                    self.ax.stability[ax_idx].axhline(self.OSSA.spike_detection_threshold, color='black', linewidth=1, zorder=50)
                if self.spike_polarity[ax_idx] == 'negative' or self.spike_polarity[ax_idx] == 'bipolar':
                    self.ax.stability[ax_idx].axhline(-self.OSSA.spike_detection_threshold, color='black', linewidth=1, zorder=50)
            # Add text box showing last selected point coordinates
            self.text.stability[ax_idx] = self.ax.stability[ax_idx].annotate('', xy=(.01, .99), xytext=(.01, .99), xycoords='axes fraction', ha='left', va='top', clip_on=True, bbox=self.OSSA.default['NoiseRemoval']['stability_plot_textbox_props'], zorder=100)

            # Feature plot
            self.lines.feature[ax_idx] = [self.ax.feature[ax_idx].plot(0, 0, alpha=1., marker='.', markersize=self.OSSA.default['NoiseRemoval']['size_bad_spikes'], linestyle='None', markeredgecolor='None', markerfacecolor=self.OSSA.default['NoiseRemoval']['color_bad_spikes'], markeredgewidth=2, picker=5, zorder=1, label=label_channel + '0')[0],
                                          self.ax.feature[ax_idx].plot(0, 0, alpha=1., marker='.', markersize=self.OSSA.default['scatter_size'], linestyle='None', markeredgecolor='None', markerfacecolor=self.OSSA.default['NoiseRemoval']['color_good_spikes'], markeredgewidth=2, picker=5, zorder=2, label=label_channel + '1')[0]]
            # Add a visible marker for the selected point
            self.last_selected_line.feature[ax_idx] = self.ax.feature[ax_idx].plot(-1000, 0, linestyle='None', marker='o', markersize=self.OSSA.default['last_selected_point_size'], markerfacecolor='r', markeredgecolor='w', markeredgewidth=1.5, zorder=3, scalex=False, scaley=False)[0]
            # Plot zero-amplitude line and detection threshold
            if self.OSSA.plotted_features['NoiseRemoval']['x']['feature'] == 'peak_amplitude_SD':
                if self.spike_polarity[ax_idx] == 'positive' or self.spike_polarity[ax_idx] == 'bipolar':
                    self.ax.feature[ax_idx].axvline(self.OSSA.spike_detection_threshold, color='black', linewidth=1, zorder=50)
                if self.spike_polarity[ax_idx] == 'negative' or self.spike_polarity[ax_idx] == 'bipolar':
                    self.ax.feature[ax_idx].axvline(-self.OSSA.spike_detection_threshold, color='black', linewidth=1, zorder=50)
            else:
                self.ax.feature[ax_idx].axvline(0, color='black', linewidth=1, zorder=50)
            if self.OSSA.plotted_features['NoiseRemoval']['y']['feature'] == 'peak_amplitude_SD':
                if self.spike_polarity[ax_idx] == 'positive' or self.spike_polarity[ax_idx] == 'bipolar':
                    self.ax.feature[ax_idx].axhline(self.OSSA.spike_detection_threshold, color='black', linewidth=1, zorder=50)
                if self.spike_polarity[ax_idx] == 'negative' or self.spike_polarity[ax_idx] == 'bipolar':
                    self.ax.feature[ax_idx].axhline(-self.OSSA.spike_detection_threshold, color='black', linewidth=1, zorder=50)
            else:
                self.ax.feature[ax_idx].axhline(0, color='black', linewidth=1, zorder=50)
            # Add text box showing firing rate of good and bad spikes
            self.text.feature[ax_idx] = self.ax.feature[ax_idx].annotate('', xy=(.01, .99), xytext=(.01, .99), xycoords='axes fraction', ha='left', va='top', clip_on=True, bbox=self.OSSA.default['NoiseRemoval']['stability_plot_textbox_props'], zorder=100)

            # Waveform plot
            colors = default_colors()  # Re-initialize for each shank
            self.lines.waveform[ax_idx] = [self.ax.waveform[ax_idx].plot(self.waveform_plot_x[ax_idx][ch], np.array([None] * self.CS.spike_waveform_duration), color=next(colors), linewidth=self.OSSA.default['line_width_thick'])[0] for ch in range(self.n_channels_these_shanks[ax_idx])]
            # Plot zero-lines
            self.ax.waveform[ax_idx].axhline(0, color='black', linewidth=1, zorder=2)
            self.ax.waveform[ax_idx].axvline(0, color='black', linewidth=1, zorder=2)
            # Add static textbox with channel name
            self.ax.waveform[ax_idx].annotate(self.CS.shank_selection_names[ax_idx], xy=(.01, .99), xytext=(.01, .99), xycoords='axes fraction', ha='left', va='top', clip_on=True, bbox=self.OSSA.default['NoiseRemoval']['stability_plot_textbox_props'])
            # Fix axes appearance
            waveform_plot_ylim = compute_array_range(self.WAVEFORMS[ax_idx], padding=True)
            self.ax.waveform[ax_idx].set_ylim(waveform_plot_ylim)
            # Fix axes appearance of waveform plot
            # Reset ticks and labels from all plots
            self.ax.waveform[ax_idx].set_xticks(list())
            self.ax.waveform[ax_idx].set_yticks(list())
            self.ax.waveform[ax_idx].set_xticklabels(list())
            # Find x-lims and apply them
            self.ax.waveform[ax_idx].set_xlim(np.hstack(self.waveform_plot_x[ax_idx]).min(), np.hstack(self.waveform_plot_x[ax_idx]).max())
            # Make appropriate x-ticks
            waveform_time = np.linspace(-self.CS.waveform_before_peak_ms, self.CS.waveform_after_peak_ms, self.CS.spike_waveform_duration)
            waveform_plot_xticklabels = np.arange(-self.CS.waveform_before_peak_ms, self.CS.waveform_after_peak_ms + .001, 0.5)
            xticks = np.array([np.abs(t - waveform_time).argmin() for t in waveform_plot_xticklabels])
            waveform_plot_xticks = np.hstack([self.waveform_plot_x[ax_idx][ch][xticks] for ch in range(self.n_channels_these_shanks[ax_idx])])
            waveform_plot_xticklabels = [waveform_plot_xticklabels] + [[''] * xticks.shape[0]] * (self.n_channels_these_shanks[ax_idx]-1)
            self.ax.waveform[ax_idx].set_xticks(np.hstack(waveform_plot_xticks))
            self.ax.waveform[ax_idx].set_xticklabels(np.hstack(waveform_plot_xticklabels))

        # Fix axes appearance of stability plot
        stability_plot_xticks_samples = np.concatenate(([0], self.CS.INFO['segments'][:, 1]))
        stability_plot_xticks_min = np.round(stability_plot_xticks_samples / self.CS.INFO['sampling_frequency'] / 60.)
        stability_plot_xticklabels = [str('%i' % i) for i in stability_plot_xticks_min]
        [i.set_xticks(stability_plot_xticks_samples) for i in self.ax.stability]
        [i.set_xticklabels(list()) for i in self.ax.stability]
        self.ax.stability[-1].set_xticklabels(stability_plot_xticklabels)
        self.ax.stability[-1].set_xlabel('Time (min)')  # hard-coded
        ylabel = translate_feature_to_axis_label(self.OSSA.plotted_features['NoiseRemoval']['y']['stability'])
        [i.set_ylabel(ylabel) for i in self.ax.stability]

        # Fix axes appearance of feature plot
        xlabel = translate_feature_to_axis_label(self.OSSA.plotted_features['NoiseRemoval']['x']['feature'])
        self.ax.feature[-1].set_xlabel(xlabel)
        ylabel = translate_feature_to_axis_label(self.OSSA.plotted_features['NoiseRemoval']['y']['feature'])
        [i.set_ylabel(ylabel) for i in self.ax.feature]

        # Apply label to last axes
        self.ax.waveform[-1].set_xlabel('Time (ms)')

        # Plot data
        self.reset_figure(reset_zoom=True)

    def reset_figure(self, reset_zoom=True):
        """Restore data in all plots"""
        # Recalculate firing rate
        self.update_suggested_spike_indices()
        self.update_firing_rates()
        # Restore plots status
        for idx in range(self.n_shanks_to_refine):
            # Get the index of the channel and of the axes
            ax_idx = self.shank_selection_table[idx, 1]
            # Stability plot
            self.update_plot('stability', ax_idx, reset_zoom)
            self.last_selected_line.stability[ax_idx].set_ydata(None)
            self.text.stability[ax_idx].set_text('')
            # Feature plot
            self.update_plot('feature', ax_idx, reset_zoom)
            self.last_selected_line.feature[ax_idx].set_ydata(None)
            self.update_panel_firing_rate(ax_idx)
            # Waveform plot
            waveforms = [self.waveform_plot_x[ax_idx][ch] * np.nan for ch in range(self.n_channels_these_shanks[ax_idx])]
            [self.lines.waveform[ax_idx][ch].set_ydata(waveforms[ch]) for ch in range(self.n_channels_these_shanks[ax_idx])]
            if reset_zoom:
                if self.use_suggestion:
                    index = np.arange(self.WAVEFORMS[ax_idx].shape[0], dtype=int)
                else:
                    index = self.DATA['suggested_good_spikes'][ax_idx]
                waveform_plot_ylim = compute_array_range(self.WAVEFORMS[ax_idx][index, :], padding=True)
                self.ax.waveform[ax_idx].set_ylim(waveform_plot_ylim)
                # Store the current y-limits
                self.ax_lims.waveform[ax_idx, 1] = waveform_plot_ylim
            self.ax.waveform[ax_idx].figure.canvas.draw_idle()
        # Reset reset_zoom
        if reset_zoom:
            self.callback_reset_zoom()


    ############################################################################
    # User interaction
    ############################################################################
    def show_GUI(self):
        """Short-hand function to show GUI"""
        self.window.showMaximized()
        self.window.raise_()
        # Change flag
        self.OSSA.currently_showing = 'NoiseRemoval'

    def on_close(self):
        # All local variables will be dropped from the main instance. The
        # only variables in which we are interested is the 'exit_code', because
        # the rest is stored on disk.
        if self.helperWindow.isVisible():
            self.helperWindow.deleteLater()
        self.window.allowed_to_close = True
        self.window.close()
        # Show the ChannelSelector GUI
        self.CS.show_GUI()

    def history_manager(self, action):
        msg = ''
        if action == 'initialize':
            initialize_field(self, 'HISTORY')
            self.HISTORY.first_state = None
            self.HISTORY.current_state = None
            initialize_field(self.HISTORY, ['undo', 'redo'], 'list', shape=self.OSSA.analysis['history_size'])
            return

        elif action == 'reset':
            msg = 'INFO: Resetting history manager to last save'
            self.DATA['suggest_spikes_array'] = backup(self.HISTORY.first_state)
            initialize_field(self.HISTORY, ['undo', 'redo'], 'list', shape=self.OSSA.analysis['history_size'])

        elif action == 'do':
            # Add latest state to the 'current' item of the history.
            if self.HISTORY.current_state is not None:
                # Add latest state to the 'undo' stack.
                self.HISTORY.undo.pop(0)
                self.HISTORY.undo.append(self.HISTORY.current_state)
            # Re-initialize the 'redo' stack if not empty
            if self.HISTORY.redo[-1] is not None:
                self.HISTORY.redo = list(np.empty((self.OSSA.analysis['history_size'],), dtype=object))

        elif action == 'undo':
            if self.HISTORY.undo[-1] is not None:  # State can be undone
                msg = 'INFO: Undo-ing last action'
                # Push current state to the 'redo' stack
                self.HISTORY.redo.pop(0)
                self.HISTORY.redo.append(self.HISTORY.current_state)
                # Move the last element of the 'undo' stack to the current state
                self.DATA['suggest_spikes_array'] = self.HISTORY.undo.pop(-1)
                # self.HISTORY.undo[-1] = backup(self.DATA['suggest_spikes_array'])
                self.HISTORY.undo.insert(0, None)
            else:
                return

        elif action == 'redo':
            if self.HISTORY.redo[-1] is not None:  # State can be redone
                msg = 'INFO: Redo-ing last action'
                # Push current state to the 'undo' stack
                self.HISTORY.undo.pop(0)
                self.HISTORY.undo.append(self.HISTORY.current_state)
                # Move the last element of the 'redo' stack to the current state
                self.DATA['suggest_spikes_array'] = self.HISTORY.redo.pop(-1)
                self.HISTORY.redo.insert(0, None)
            else:
                return

        # Log message
        if action != 'do':
            LOGGER.info(msg)

        # Update current state before it gets modified externally
        self.HISTORY.current_state = backup(self.DATA['suggest_spikes_array'])
        # Update plots
        self.reset_figure(reset_zoom=False)

    def toggle_suggest_button(self, state):
        """What happens when toggle_suggest button is pressed"""
        if state:
            self.menu.toolbar_suggest.setIcon(QtGui.QIcon(QtGui.QPixmap(self.OSSA.default['icon_visible'])))
        else:
            self.menu.toolbar_suggest.setIcon(QtGui.QIcon(QtGui.QPixmap(self.OSSA.default['icon_invisible'])))
        self.use_suggestion = state
        # Toggle toolbar button
        self.menu.toolbar_suggest.setChecked(state)

    def toggle_button_pan_zoom(self, toolbar_state):
        """It happens when button to pan/zoom is toggled."""
        # Get the current state
        current_state = self.currently_panning
        # Continue only if state changed
        if current_state != toolbar_state:
            # Toggle state variable
            self.currently_panning = toolbar_state
            # Toggle panning
            for plt_name in self._plot_names:
                for toolbar in self.matplotlib_toolbar[plt_name]:
                    toolbar.pan()
        # Toggle toolbar button
        self.menu.toolbar_pan_zoom.setChecked(toolbar_state)

    def update_plot(self, plot_name, selected_axes, zoom=True):
        """Method to update plots."""
        # Selected points
        index = self.DATA['suggested_good_spikes'][selected_axes]
        x = self.DATA['%s_x' % plot_name][selected_axes][index]
        y = self.DATA['%s_y' % plot_name][selected_axes][index]
        self.lines[plot_name][selected_axes][1].set_xdata(x)
        self.lines[plot_name][selected_axes][1].set_ydata(y)
        # Unselected points
        if self.use_suggestion:
            # Draw points
            index = self.DATA['suggested_bad_spikes'][selected_axes]
            x = self.DATA['%s_x' % plot_name][selected_axes][index]
            y = self.DATA['%s_y' % plot_name][selected_axes][index]
            self.lines[plot_name][selected_axes][0].set_xdata(x)
            self.lines[plot_name][selected_axes][0].set_ydata(y)
            # Compute axes limits on all points
            x = self.DATA['%s_x' % plot_name][selected_axes]
            y = self.DATA['%s_y' % plot_name][selected_axes]
        else:
            # Draw points
            self.lines[plot_name][selected_axes][0].set_xdata(None)
            self.lines[plot_name][selected_axes][0].set_ydata(None)
        # Fix figure limits
        if zoom:
            self.set_plot_lims(x, y, selected_axes, from_plot=plot_name)
        # Update canvas
        self.ax[plot_name][selected_axes].figure.canvas.draw()

    def update_last_selected_point(self, selected_axes, x_stability, y_stability, x_feature, y_feature, color):
        """Method to update only the last selected point handle."""
        self.last_selected_line.stability[selected_axes].set_xdata(x_stability)
        self.last_selected_line.stability[selected_axes].set_ydata(y_stability)
        self.last_selected_line.stability[selected_axes].set_markerfacecolor(color)
        self.last_selected_line.feature[selected_axes].set_xdata(x_feature)
        self.last_selected_line.feature[selected_axes].set_ydata(y_feature)
        self.last_selected_line.feature[selected_axes].set_markerfacecolor(color)

    def set_plot_lims(self, x, y, selected_axes, from_plot):
        """Method to set and store the axes limits for any plot."""
        if from_plot == 'waveform':
            add_padding = False
        else:
            add_padding = True
        x_lims = compute_array_range(x, add_padding)
        y_lims = compute_array_range(y, padding=True)
        # Save limits
        self.ax_lims[from_plot][selected_axes, :] = [x_lims, y_lims]
        # Reset axes limits
        self.ax[from_plot][selected_axes].set_xlim(self.ax_lims[from_plot][selected_axes, 0])
        self.ax[from_plot][selected_axes].set_ylim(self.ax_lims[from_plot][selected_axes, 1])
        self.ax[from_plot][selected_axes].figure.canvas.draw_idle()

    def start_polygon_selector(self, discard_or_keep, from_keyboard):
        """Routine to pass currently visible points to the polygon selector, and
        activate user interaction with it."""
        self.toggle_button_pan_zoom(False)
        # Disable the other toolbar button used for selection. Also, instead of
        # passing all the points to the selector, only pass those that can be
        # changed by the selection, that is, the 'good' spikes when we want to
        # discard and the 'bad' spikes when we want to keep.
        if discard_or_keep == 'discard':
            self.menu.select_to_keep.setEnabled(False)
            self.menu.select_to_keep.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(self.OSSA.default['NoiseRemoval']['channel_button_color_neutral'], dtype=float)*255))))
            color = self.OSSA.default['NoiseRemoval']['selector_color_discard']
        else:
            self.menu.select_to_discard.setEnabled(False)
            self.menu.select_to_discard.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(self.OSSA.default['NoiseRemoval']['channel_button_color_neutral'], dtype=float)*255))))
            color = self.OSSA.default['NoiseRemoval']['selector_color_keep']
        # Mark the current state
        self.currently_selecting_spikes = True
        # Get data
        for idx in range(self.n_shanks_to_refine):
            # Get the index of the channel and of the axes
            ax_idx = self.shank_selection_table[idx, 1]
            # Get indices of the points
            if discard_or_keep == 'discard':
                index = self.DATA['suggested_good_spikes'][ax_idx]
            else:
                index = self.DATA['suggested_bad_spikes'][ax_idx]
            # Get the actual values
            x_stability = self.DATA['stability_x'][ax_idx][index]
            y_stability = self.DATA['stability_y'][ax_idx][index]
            x_feature = self.DATA['feature_x'][ax_idx][index]
            y_feature = self.DATA['feature_y'][ax_idx][index]
            # Activate polygon selector
            self.polygon_selectors.stability.append(SelectFromCollection(self.ax.stability[ax_idx], x_stability, y_stability, index, color, partial(self.validate_polygon_selection, discard_or_keep=discard_or_keep, from_keyboard=from_keyboard)))
            self.polygon_selectors.feature.append(SelectFromCollection(self.ax.feature[ax_idx], x_feature, y_feature, index, color, partial(self.validate_polygon_selection, discard_or_keep=discard_or_keep, from_keyboard=from_keyboard)))

    def validate_polygon_selection(self, discard_or_keep, from_keyboard):
        """Callback to accept the spike selection and update data and plots."""
        # Continue only if the exit code from the polygon selectors was not equal to 0
        # return
        if not np.any(np.concatenate(([i.exit_code for i in self.polygon_selectors.stability], [i.exit_code for i in self.polygon_selectors.feature])) == 0):
            valid_selection = True
            # If polygon is not closed, clicking doesn't have any effect
            if not any(([i.polygon_completed for i in self.polygon_selectors.stability], [i.polygon_completed for i in self.polygon_selectors.feature])):
                valid_selection = False
            # Find indices of selected points
            INDICES, any_spike_selected = self.get_indices_of_points_in_selector()
            if not any_spike_selected:
                valid_selection = False
            if valid_selection:
                # Change spike type in data
                for idx in range(self.n_shanks_to_refine):
                    # Get the index of the channel and of the axes
                    ax_idx = self.shank_selection_table[idx, 1]
                    # Check whether any spike was selected on this channel
                    if INDICES[ax_idx].shape[0] == 0:
                        continue
                    if discard_or_keep == 'discard':
                        self.DATA['suggest_spikes_array'][ax_idx][INDICES[ax_idx]] = False
                    else:
                        self.DATA['suggest_spikes_array'][ax_idx][INDICES[ax_idx]] = True
                # Update history
                self.history_manager(action='do')
                msg = 'Accepted selection'
                # A new selection will start
                if from_keyboard:
                    restart_selection = False
                else:
                    restart_selection = True
            else:
                restart_selection = False
                msg = '(canceled polygon)'
            self.show_in_console_and_statusbar(msg)
        else:
            restart_selection = False
            msg = '(canceled polygon)'
            self.show_in_console_and_statusbar(msg)

        # Stop polygon selector
        for ii in range(len(self.polygon_selectors.stability)):
            # Disconnect the polygon selector and reactivate the waveform selector
            self.polygon_selectors.stability[ii].disconnect()
            self.polygon_selectors.feature[ii].disconnect()
        # Remove selectors
        initialize_field(self.polygon_selectors, ['stability', 'feature'], 'list')
        self.currently_selecting_spikes = False

        # Re-enable buttons
        self.menu.select_to_keep.setEnabled(True)
        self.menu.select_to_discard.setEnabled(True)
        self.menu.select_to_keep.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(self.OSSA.default['NoiseRemoval']['channel_button_color_keep'], dtype=float)*255))))
        self.menu.select_to_discard.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(self.OSSA.default['NoiseRemoval']['channel_button_color_discard'], dtype=float)*255))))
        # Start a new selection
        if restart_selection:
            self.start_polygon_selector(discard_or_keep, from_keyboard)

    def get_indices_of_points_in_selector(self):
        """Get the index of the points enclosed in the polygons of feature and
        stability plots."""
        any_spike_selected = False
        INDICES = list(np.empty((self.n_shanks_to_refine,), dtype=object))
        for idx in range(self.n_shanks_to_refine):
            # Get the union of the indices of the selected points
            points_selected_on_stability_plot = self.polygon_selectors.stability[idx].ind
            points_selected_on_feature_plot = self.polygon_selectors.feature[idx].ind
            relative_ind = np.union1d(points_selected_on_stability_plot, points_selected_on_feature_plot).astype(int)
            # These relative indices are now transformed in their position in the DATA
            ind = self.polygon_selectors.stability[idx].points_id[relative_ind].astype(int)
            # If at least one spike was selected, raise a flag
            if ind.shape[0] > 0 and not any_spike_selected:
                any_spike_selected = True
            # Concatenate data
            INDICES[idx] = ind
        return INDICES, any_spike_selected

    def update_panel_firing_rate(self, selected_axes):
        """Method to update the panel showing the firing rate in the feature plot."""
        text = '%.1f' % self.DATA['firing_rate'][selected_axes][0]
        if self.use_suggestion:
            text += '\n%.1f' % self.DATA['firing_rate'][selected_axes][1]
        self.text.feature[selected_axes].set_text(text)
        self.ax.feature[selected_axes].figure.canvas.draw_idle()


    ############################################################################
    # Callback functions
    ############################################################################
    def callback_on_resize(self):
        for plt_name in self._plot_names:
            for fig in self.fig[plt_name]:
                try:
                    fig.tight_layout(pad=0)
                except ValueError:  # This happens when the axes is squeezed too much and a new axes-range cannot be computed
                    pass  # Silently ignore it
            for ax in self.ax[plt_name]:
                ax.figure.canvas.draw_idle()

    def callback_show_clicked_waveform(self, event, from_plot):
        """Callback to show and update the waveform of the clicked spike."""
        # Check whether waveform picking is allowed
        if not self.is_waveform_picking_allowed(event.mouseevent.inaxes, from_plot):
            return
        # Get number of points around cursor and keep only first one
        N_points = len(event.ind)
        if not N_points:  # If no points clicked (==None), then return immediately
            return
        # Find out which axis the user clicked on, and which channel these data come from
        selected_shank, selected_axes, selecting_good_spike = self._get_clicked_plot_info(event.artist)
        if selected_shank is None:
            return
        # Get a random point among selected
        data_point_index = event.ind[np.random.choice(N_points)]
        # Get index of point in data (good + bad)
        if selecting_good_spike:
            real_data_point_index = self.DATA['suggested_good_spikes'][selected_axes][data_point_index]
            color = self.OSSA.default['NoiseRemoval']['color_good_spikes']
        else:
            real_data_point_index = self.DATA['suggested_bad_spikes'][selected_axes][data_point_index]
            color = self.OSSA.default['NoiseRemoval']['color_bad_spikes']
        # Update waveform plot data and color
        wf = self.WAVEFORMS[selected_axes][real_data_point_index, :, :].transpose().tolist()
        [self.lines.waveform[selected_axes][ch].set_ydata(wf[ch]) for ch in range(self.n_channels_these_shanks[selected_axes])]
        # Update limits in all plots
        all_wf_data = np.hstack([[self.lines.waveform[ax_idx][ch].get_ydata() for ch in range(self.n_channels_these_shanks[ax_idx])] for ax_idx in range(self.n_shanks_to_refine)])
        all_wf_data = all_wf_data[np.logical_not(np.isnan(all_wf_data))]
        if all_wf_data.shape[0] > 0:
           ylims = compute_array_range(all_wf_data, padding=True)
           [self.ax.waveform[ax_idx].set_ylim(ylims) for ax_idx in range(self.n_shanks_to_refine)]

        # Get selected point position
        x_stability = self.DATA['stability_x'][selected_axes][real_data_point_index]
        x_stability_relative = self.DATA['time'][selected_axes][real_data_point_index]
        y_stability = self.DATA['stability_y'][selected_axes][real_data_point_index]
        x_feature = self.DATA['feature_x'][selected_axes][real_data_point_index]
        y_feature = self.DATA['feature_y'][selected_axes][real_data_point_index]
        self.update_last_selected_point(selected_axes, x_stability, y_stability, x_feature, y_feature, color)
        # Update info on selected point in stability plot
        tooltip_base = translate_feature_list_to_tooltip(['timestamp', self.OSSA.plotted_features['NoiseRemoval']['y']['stability'], self.OSSA.plotted_features['NoiseRemoval']['x']['feature'], self.OSSA.plotted_features['NoiseRemoval']['y']['feature']])
        self.text.stability[selected_axes].set_text(tooltip_base % (x_stability_relative, y_stability, x_feature, y_feature))
        # Update figures
        for plt_name in self._plot_names:
            self.ax[plt_name][selected_axes].figure.canvas.draw_idle()

    def callback_toggle_suggestions(self):
        """Callback to activate / inactivate spike type suggestions."""
        prev_state = self.use_suggestion
        new_state = not prev_state
        if new_state:
            msg = 'Discarded spikes are visible'
        else:
            msg = 'Discarded spikes are invisible'
        self.show_in_console_and_statusbar(msg)
        self.toggle_suggest_button(new_state)
        self.reset_figure(reset_zoom=False)
        self.show_in_console_and_statusbar('ok')

    def callback_pan_zoom_figure(self):
        """Callback to toggle pan / zoom of the figures"""
        self.toggle_button_pan_zoom(not self.currently_panning)

    def callback_reset_zoom(self):
        """Callback to reset the zoom of all plots"""
        for ax_idx in range(self.n_shanks_to_refine):
            self.update_plot('stability', ax_idx, zoom=True)
            self.update_plot('feature', ax_idx, zoom=True)
            if self.use_suggestion:
                index = np.arange(self.WAVEFORMS[ax_idx].shape[0], dtype=int)
            else:
                index = self.DATA['suggested_good_spikes'][ax_idx]
            waveform_plot_ylim = compute_array_range(self.WAVEFORMS[ax_idx][index, :], padding=True)
            self.ax.waveform[ax_idx].set_ylim(waveform_plot_ylim)
            # Store the current y-limits
            self.ax_lims.waveform[ax_idx, 1] = waveform_plot_ylim
            self.ax.waveform[ax_idx].figure.canvas.draw_idle()

    def callback_quit(self, save):
        """Callback to cancel any current change and show the ChannelSelector GUI."""
        if save:  # Save current state of the user selection
            LOGGER.info('Saving data to disk')
            self.update_suggested_spike_indices()
            for shank_idx, shank_name in enumerate(self.CS.shank_selection_names):
                # Replace the 'good spike' feature on disk with the results of this GUI
                update_hdf5_data(self.CS.spikes_hdf5_filename, shank_name, column='good_spike', values=self.DATA['suggest_spikes_array'][shank_idx])
                # Check whether there are 'good' spikes
                contains_spikes = self.DATA['suggested_good_spikes'][shank_idx].shape[0] > 0
                # Update OSSA_results_table
                self.CS.update_OSSA_results(shank_name, 'NoiseRemoval', contains_spikes)

            LOGGER.info('ok', append=True)
            # Assign exit code
            self.OSSA.exit_code = 1
        else:
            self.OSSA.exit_code = 0
        # Close this GUI
        LOGGER.info('Quitting GUI and going back to ChannelSelector')
        self.on_close()
        LOGGER.info('ok', append=True)

    def callback_reload(self):
        self.history_manager(action='reset')

    def callback_select_to_discard(self, from_keyboard=False):
        """Callback to select and discard spikes from all channels."""
        if self.menu.select_to_discard.isEnabled():
            if self.currently_selecting_spikes:
                if not from_keyboard:
                    self.validate_polygon_selection('discard', from_keyboard)
                else:  # Ignore invoking keyboard shortcut again. Use appropriate command to terminate the selection.
                    pass
            else:
                self.show_in_console_and_statusbar('Selecting spikes to discard')
                self.start_polygon_selector('discard', from_keyboard)

    def callback_select_to_keep(self, from_keyboard=False):
        """Callback to select and keep spikes from all channels."""
        if self.menu.select_to_keep.isEnabled():
            if self.currently_selecting_spikes:
                if not from_keyboard:
                    self.validate_polygon_selection('keep', from_keyboard)
                else:  # Ignore invoking keyboard shortcut again. Use appropriate command to terminate the selection.
                    pass
            else:
                # Before starting the selection, we have to re-enable the suggestion
                # button and make the 'noisy' events appear, so we can select
                # from them.
                self.toggle_suggest_button(True)
                self.reset_figure(reset_zoom=False)
                # Start selector
                msg = 'Selecting spikes to keep'
                self.show_in_console_and_statusbar(msg)
                self.start_polygon_selector('keep', from_keyboard)

    def callback_show_help(self):
        self.helperWindow.on_show()
        self.show_in_console_and_statusbar('Showing helper')


    ############################################################################
    # Misc methods
    ############################################################################
    @staticmethod
    def _get_clicked_plot_info(artist):
        """Extract channel and axes index from the label written in the line2D
        handle."""
        selected_plot_label = artist.get_label()
        if selected_plot_label == '':  # The selected graphic doesn't have a label
            return
        index_of_underscore = [i for i in range(len(selected_plot_label)) if selected_plot_label.startswith('_', i)]
        selected_shank = int(selected_plot_label[:index_of_underscore[0]])
        selected_axes = int(selected_plot_label[index_of_underscore[0] + 1:index_of_underscore[1]])
        selecting_good_spike = int(selected_plot_label[index_of_underscore[1] + 1:]) == 1
        return selected_shank, selected_axes, selecting_good_spike

    def show_in_console_and_statusbar(self, msg):
        LOGGER.info(msg)
        self.menu.statusBar.showMessage(msg)

    def is_waveform_picking_allowed(self, ax, from_plot):
        # No, if the panning tool is on
        if self.currently_panning:
            return False
        # Yes, if the polygon selector is not active
        if not self.currently_selecting_spikes:
            return True
        # If the polygon selector is on, check whether the polygon on the current
        # axes is still active for drawing
        for p in self.polygon_selectors[from_plot]:
            if p.ax == ax:
                if p.currently_drawing_polygon:
                    return False
                else:
                    return True
