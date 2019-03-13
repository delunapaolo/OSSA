# -*- coding: utf-8 -*-
"""
GUI for refinement of spike clusters.
"""
# System
import re
from functools import partial
from copy import deepcopy as backup

# Graphics
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

# Numerical computation
import numpy as np
from scipy.linalg import svd
import pandas as pd
from itertools import chain

# Workflow
from Code.general_configs import GC
from Code.IO_operations.log4py import LOGGER
from Code.GUIs.GUI_configs import default_colors
from Code.GUIs.GUI_utilities import initialize_field
from Code.Workflows.Spike_sorting_workflow.OSSA.analysis_functions import correlogram, correlogram_make_lag_axis, compute_array_range, divide0
from Code.Workflows.Spike_sorting_workflow.OSSA.visualization import OSSA_window, OSSAFigureCanvas, OSSAHelpDialog, make_QSplitter, SelectFromCollection, color_Qt_button, translate_feature_to_axis_label, set_figure_style, translate_feature_list_to_tooltip, QRangeSlider
from Code.IO_operations.spikes_hdf5 import read_hdf5_data, read_waveforms, update_hdf5_data, write_sorting_summary_table, read_sorting_summary_table, get_list_of_all_cluster_ids


################################################################################
# GUI
################################################################################
class OSSA_SpikeSorter_GUI(object):
    def __init__(self, ChannelSelector):
        LOGGER.info('SpikeSorter GUI', decorate=True)

        # Reference the OSSA and ChannelSelector instances at the root level.
        self.OSSA = ChannelSelector.OSSA
        self.CS = ChannelSelector

        # Unpack default values for correlogram computation
        self.CCG_max_lag = self.OSSA.analysis['ACG_max_lag']
        self.CCG_bin_size = self.OSSA.analysis['ACG_bin_size']

        # Initialize fields that will contain graphical handles
        self.fig = None
        self.matplotlib_canvas = None
        self.matplotlib_toolbar = None
        self.ax = None
        self.CLUSTER_COLORS = default_colors()
        self.ax_lims = None
        self.text = None
        self.x_data = None
        self.y_data = None
        self.last_selected_line = None
        self.menu = None
        self.keyboard = None
        self.textEdits = None
        self.callbacks = None
        self.lambdas = None
        # Callback variables
        self.waveform_picker_tolerance = 5  # pixels around a data-point to trigger the callback
        self.is_saved = False
        self.currently_panning = False
        self.currently_selected_clusters = np.empty((0, ), dtype=int)
        self.currently_selecting_spikes = False
        self.n_channels_this_shank = self.CS.n_channels_electrode[self.CS.shank_selection_index]
        # Initialize tables to keep data in memory
        self.DATA = None
        self.n_clusters = 0
        self.HISTORY = None
        self.SUMMARY_TABLE = None
        self.correlograms = None
        self.QTable = None
        self.feature_time_range_samples = None

        # Make the helper message for keyboard shortcuts
        shortcuts = """
        F1\tHelp
        %s\tSave and close
        %s\tDiscard and close
        %s\tReload last save
        %s / %s\tUndo / Redo (up to %i times)
        
        %s\tSelect to add cluster
        %s\tRemove clusters
        %s\tSelect to uncluster spikes
        %s\tMerge clusters
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
               self.OSSA.keyboard_shortcuts['SpikeSorter']['add_cluster'],
               self.OSSA.keyboard_shortcuts['SpikeSorter']['delete_cluster'],
               self.OSSA.keyboard_shortcuts['SpikeSorter']['edit_cluster'],
               self.OSSA.keyboard_shortcuts['SpikeSorter']['merge_clusters'],
               self.OSSA.keyboard_shortcuts['pan_zoom'],
               self.OSSA.keyboard_shortcuts['reset_zoom'])

        # Make windows
        self.window = OSSA_window(title='%s - shank %s' % (self.OSSA.animal_ID, self.CS.shank_selection_names))
        self.helperWindow = OSSAHelpDialog(shortcuts)

        # Initialize GUI data and history manager
        self._initialize_handles()
        self.history_manager(action='initialize')

        # Initialize data
        self._initialize_GUI_data()
        self._initialize_correlograms()

        # Make GUI
        self._GUI_create_main_frame()
        self.initialize_plots(include_selected_point=True)
        self._cleanup_init()

        # Initialize all clusters as currently selected
        self.currently_selected_clusters = self.SUMMARY_TABLE['id'].values
        self._table_focus_on_cluster(self.currently_selected_clusters)

        # Update current state of history manager with latest data
        self.HISTORY.first_state[1] = backup(self.correlograms)
        self.HISTORY.current_state[1] = backup(self.correlograms)
        self.HISTORY.first_state[2] = backup(self.SUMMARY_TABLE)
        self.HISTORY.current_state[2] = backup(self.SUMMARY_TABLE)

        LOGGER.info('ok', append=True)

        # This GUI does not show itself, but it's initialized by the ChannelSelector
        # GUI and made visible from there.


    def _cleanup_init(self):
        """Routines to be performed at the end of __init__ function."""
        # Destroy references to matplotlib's figures. These are now referenced
        # by the Qt event loop and don't need to be opened by the matplotlib's
        # renderer.
        for plt_name in self._plot_names_apart_waveform:
            plt.close(self.fig[plt_name])
        [plt.close(self.fig['waveform'][ch]) for ch in range(self.n_channels_this_shank)]

    def _initialize_handles(self):
        """Create all the attributes that will hold graphical elements"""
        # These are the names of the plots. If a new one needs to be added, this
        # is the place
        self._plot_names_apart_waveform = ['stability', 'feature', 'correlogram']
        self._plot_names = self._plot_names_apart_waveform + ['waveform']
        plots_responding_to_user_selection = ['stability', 'feature', 'waveform']
        # Names of buttons
        self._button_names_plot = ['add_cluster', 'delete_cluster', 'edit_cluster', 'merge_clusters', 'compute_ISI', 'force_clean_ISI', 'compute_PC', 'compute_CCG']
        self._button_text_edit = ['edit_ACG_bin', 'edit_ACG_lag']

        # Figures
        initialize_field(self, 'fig')
        initialize_field(self.fig, self._plot_names, 'list')
        initialize_field(self, 'matplotlib_canvas')
        initialize_field(self.matplotlib_canvas, self._plot_names, 'list')
        initialize_field(self, 'matplotlib_toolbar')
        initialize_field(self.matplotlib_toolbar, self._plot_names, 'list')
        initialize_field(self, 'ax')
        initialize_field(self.ax, self._plot_names, 'list')
        # Plots
        initialize_field(self, 'last_selected_line')
        initialize_field(self.last_selected_line, plots_responding_to_user_selection, '=', None)
        initialize_field(self, 'x_data')
        initialize_field(self.x_data, self._plot_names)
        initialize_field(self, 'y_data')
        initialize_field(self.y_data, self._plot_names)
        initialize_field(self, 'ax_lims')
        initialize_field(self.ax_lims, self._plot_names, 'list', shape=2)  # 2 dims: x_lims and y_lims
        initialize_field(self, 'text')
        initialize_field(self.text, 'stability')
        # Buttons
        initialize_field(self, 'textEdits')
        initialize_field(self.textEdits, self._button_text_edit, '=', None)
        initialize_field(self, 'menu')
        initialize_field(self, 'keyboard')
        # Callbacks
        initialize_field(self, 'callbacks')
        initialize_field(self.callbacks, ['pick_spike', 'polygon_selectors', 'CCG'])
        self.callbacks.pick_spike = np.empty((2, ), dtype=object).tolist()  # 2 plots: stability + feature
        initialize_field(self.callbacks.polygon_selectors, ['stability', 'feature'], 'list')

    def _initialize_GUI_data(self):
        """Create tables and arrays to hold data internally."""
        # First, load data from disk
        self._load_data_of_shank()

        # Reset sorting summary table
        self.reset_summary_table(from_disk=False, store_to_disk=True)

        # Pre-allocate waveform data
        self.x_data['waveform'] = np.arange(self.CS.spike_waveform_duration)
        self.y_data['waveform'] = [np.zeros_like(self.x_data['waveform'], dtype=float) for _ in range(self.n_channels_this_shank)]

        # Make a look-up table to switch from data-points to ms of the waveform
        self.waveform_sample_time = np.linspace(-self.CS.waveform_before_peak_ms, self.CS.waveform_after_peak_ms, self.CS.spike_waveform_duration)
        self.feature_time_range_ms = [list(GC.spike_waveform_duration) for _ in range(self.n_channels_this_shank)]
        self.feature_time_range_samples = [np.arange(self.CS.spike_waveform_duration) for _ in range(self.n_channels_this_shank)]

    def _load_data_of_shank(self):
        """Load data of channels of interest. Also, suggest events that might not
        be good as spikes.
        """
        # Read .waveforms files
        self.WAVEFORMS = read_waveforms(self.CS.spikes_hdf5_filename, self.CS.shank_selection_names)
        # Get spike features from spikes previously marked as 'good'
        data = read_hdf5_data(self.CS.spikes_hdf5_filename, self.CS.shank_selection_names, where='good_spike == True', return_as_pandas=True)
        # spikes_timestamps, peak_amplitude, peak_amplitude_SD, peak_to_trough_mV = self._get_data(ch_name, column=1)
        data['trough_amplitude'] = (data['peak_to_trough_mV'] - np.abs(data['peak_amplitude'])) * -np.sign(data['peak_amplitude'])
        # Get variability of baseline
        data['variability_baseline'] = divide0(data['peak_amplitude'], data['peak_amplitude_SD'], 0)
        # Calculate the trough amplitude in SD units
        data['trough_amplitude_SD'] = divide0(data['trough_amplitude'], data['variability_baseline'], 0)
        # Keep in memory the initial state of the data
        self.HISTORY.first_state[0] = data['cluster_id_manual'].values
        # Remove unused columns
        del data['peak_amplitude']
        del data['trough_amplitude']
        del data['cluster_id_auto']
        del data['variability_baseline']

        # Store the rest of the data in memory
        self.DATA = data
        # TAke a snapshot of the current state of the data
        self.HISTORY.current_state[0] = backup(self.DATA['cluster_id_manual'].values)

    def _initialize_correlograms(self):
        self._reset_correlogram_xy_data()
        # Compute ACGs
        self.correlograms = pd.DataFrame(columns=['cluster_1', 'cluster_2', 'correlation', 'updated'])
        self.compute_all_ACGs()

    def _GUI_create_main_frame(self):
        # Create empty widget
        self.qFrame = QtWidgets.QWidget(parent=self.window)
        # Make the background white
        p = self.qFrame.palette()
        p.setColor(self.qFrame.backgroundRole(), QtCore.Qt.white)
        self.qFrame.setPalette(p)
        self.qFrame.setAutoFillBackground(True)

        # Make table widget
        self.QTable = MultiColumn_QTable(self.SUMMARY_TABLE.copy(), self.OSSA.default['widget_minimum_size'])

        # Make plots
        set_figure_style(self.OSSA.default['axes_color'])
        for plt_name in self._plot_names_apart_waveform:
            # Open figure with an axes in it
            self.fig[plt_name] = Figure(facecolor='w')
            self.ax[plt_name] = self.fig[plt_name].add_subplot(1, 1, 1)
            # Copy figure to a FigureCanvas, and add a hidden toolbar (controlled
            # by callback functions)
            self.matplotlib_canvas[plt_name] = OSSAFigureCanvas(self.fig[plt_name], self.qFrame)
            # Create hidden toolbar for access to panning tool
            self.matplotlib_toolbar[plt_name] = NavigationToolbar(self.matplotlib_canvas[plt_name], self.qFrame)
            self.matplotlib_toolbar[plt_name].hide()

            # Change resize policy of the graph to limit the minimum size
            self.matplotlib_canvas[plt_name].setMinimumSize(self.OSSA.default['widget_minimum_size'], self.OSSA.default['widget_minimum_size'])
            self.matplotlib_canvas[plt_name].updateGeometry()

        # Make toolbar's actions
        self.menu.add_cluster = QtWidgets.QToolButton(self.window)
        self.menu.add_cluster.setIcon(QtGui.QIcon(QtGui.QPixmap(self.OSSA.default['icon_add'])))
        self.menu.add_cluster.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.menu.add_cluster.setCheckable(False)
        self.menu.add_cluster.setToolTip('Enable selector of spikes to make a new cluster')
        self.menu.add_cluster.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(self.OSSA.default['SpikeSorter']['button_color_add'], dtype=float)*255))))
        self.menu.edit_cluster = QtWidgets.QToolButton(self.window)
        self.menu.edit_cluster.setIcon(QtGui.QIcon(QtGui.QPixmap(self.OSSA.default['icon_edit'])))
        self.menu.edit_cluster.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.menu.edit_cluster.setCheckable(False)
        self.menu.edit_cluster.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(self.OSSA.default['SpikeSorter']['button_color_delete'], dtype=float)*255))))
        self.menu.edit_cluster.setToolTip('Enable selector of spikes to uncluster (move to cluster 0)')
        self.menu.delete_cluster = QtWidgets.QToolButton(self.window)
        self.menu.delete_cluster.setIcon(QtGui.QIcon(QtGui.QPixmap(self.OSSA.default['icon_delete'])))
        self.menu.delete_cluster.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.menu.delete_cluster.setCheckable(False)
        self.menu.delete_cluster.setToolTip('Remove clusters (all spikes are moved to cluster 0)')
        self.menu.merge_clusters = QtWidgets.QToolButton(self.window)
        self.menu.merge_clusters.setIcon(QtGui.QIcon(QtGui.QPixmap(self.OSSA.default['icon_merge'])))
        self.menu.merge_clusters.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.menu.merge_clusters.setCheckable(False)
        self.menu.merge_clusters.setToolTip('Merge clusters (all spikes are moved to larger cluster)')
        self.menu.toolbar_pan_zoom = QtWidgets.QToolButton(self.window)
        self.menu.toolbar_pan_zoom.setIcon(QtGui.QIcon(QtGui.QPixmap(self.OSSA.default['icon_magnifying_glass'])))
        self.menu.toolbar_pan_zoom.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.menu.toolbar_pan_zoom.setCheckable(True)
        self.menu.toolbar_pan_zoom.setToolTip('Toggle pan/zoom tool in all plots')
        self.menu.toolbar_reset_zoom = QtWidgets.QToolButton(self.window)
        self.menu.toolbar_reset_zoom.setIcon(QtGui.QIcon(QtGui.QPixmap(self.OSSA.default['icon_home'])))
        self.menu.toolbar_reset_zoom.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.menu.toolbar_reset_zoom.setCheckable(False)
        self.menu.toolbar_pan_zoom.setToolTip('Toggle spike suggestions')
        self.menu.compute_ISI = QtWidgets.QToolButton(self.window)
        self.menu.compute_ISI.setText('ISI')
        self.menu.compute_ISI.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.menu.compute_ISI.setCheckable(False)
        self.menu.compute_ISI.setFixedSize(QtCore.QSize(32, 32))
        self.menu.compute_ISI.setToolTip('Show Inter-Spike Interval distribution')
        self.menu.force_clean_ISI = QtWidgets.QToolButton(self.window)
        self.menu.force_clean_ISI.setText('clean ISI')
        self.menu.force_clean_ISI.setIcon(QtGui.QIcon(QtGui.QPixmap(self.OSSA.default['icon_clean'])))
        self.menu.force_clean_ISI.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.menu.force_clean_ISI.setCheckable(False)
        # self.menu.force_clean_ISI.setFixedSize(QtCore.QSize(32, 32))
        self.menu.force_clean_ISI.setToolTip('Move the spikes that violate the refractory period\nof the selected cluster to other clusters')
        self.menu.compute_CCG = QtWidgets.QToolButton(self.window)
        self.menu.compute_CCG.setText('CCG')
        self.menu.compute_CCG.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.menu.compute_CCG.setCheckable(False)
        self.menu.compute_CCG.setFixedSize(QtCore.QSize(32, 32))
        self.menu.compute_CCG.setToolTip('Show Auto- and Cross-Correlograms')
        self.menu.compute_PC = QtWidgets.QToolButton(self.window)
        self.menu.compute_PC.setText('PC')
        self.menu.compute_PC.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.menu.compute_PC.setCheckable(False)
        self.menu.compute_PC.setFixedSize(QtCore.QSize(32, 32))
        self.menu.compute_PC.setToolTip('Recompute Principal Components of selected clusters')

        # Make text fields to edit bin-size and lag of ACG
        self.textEdits.edit_ACG_bin = QtWidgets.QLineEdit()
        self.textEdits.edit_ACG_bin.setAlignment(QtCore.Qt.AlignRight)
        input_validator = QtGui.QIntValidator()
        input_validator.setBottom(1)
        self.textEdits.edit_ACG_bin.setValidator(input_validator)  # Make sure input is a number
        self.textEdits.edit_ACG_bin.setToolTip('Time bin to calculate the Auto-Correlogram (integer value > 0)')
        self.textEdits.edit_ACG_lag = QtWidgets.QLineEdit()
        self.textEdits.edit_ACG_lag.setAlignment(QtCore.Qt.AlignRight)
        input_validator = QtGui.QIntValidator()
        input_validator.setBottom(1)
        self.textEdits.edit_ACG_lag.setValidator(input_validator)  # Make sure input is a integer
        self.textEdits.edit_ACG_lag.setToolTip('Maximum time lag at which to calculate the Auto-Correlogram (integer value > 0)')

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
        # Add an action for each waveform feature that could plotted
        features = list(self.DATA.columns)
        # Remove the features that cannot be plotted
        features.remove('segment')
        features.remove('timestamp')
        features.remove('time')
        features.remove('cluster_id_manual')
        features.remove('good_spike')
        if 'index' in features:
            features.remove('index')
        # Sort features alphabetically
        features = self.natural_sort(features)
        initialize_field(self.menu, features, 'list', shape=3)
        for f in features:
            label = translate_feature_to_axis_label(f)
            for ii in range(3):
                self.menu[f][ii] = QtWidgets.QAction(label, self.window)
                self.menu[f][ii].setCheckable(True)
        # Make help menu
        self.menu.help = QtWidgets.QAction('Help', self.window)
        self.menu.help.setShortcut('F1')

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
        # Stability y-axis
        self.menu.stability_y = QtWidgets.QMenu('Stability (y)', self.window)
        self.menu.stability_y_actions = QtWidgets.QActionGroup(self.window)
        for f in features:
            act = self.menu.stability_y_actions.addAction(self.menu[f][0])
            self.menu.stability_y.addAction(act)
        self.menu.menubar.addMenu(self.menu.stability_y)
        # Feature x-axis
        self.menu.feature_x = QtWidgets.QMenu('Feature (x)', self.window)
        self.menu.feature_x_actions = QtWidgets.QActionGroup(self.window)
        for f in features:
            act = self.menu.feature_x_actions.addAction(self.menu[f][1])
            self.menu.feature_x.addAction(act)
        self.menu.menubar.addMenu(self.menu.feature_x)
        # Feature y-axis
        self.menu.feature_y = QtWidgets.QMenu('Feature (y)', self.window)
        self.menu.feature_y_actions = QtWidgets.QActionGroup(self.window)
        for f in features:
            act = self.menu.feature_y_actions.addAction(self.menu[f][2])
            self.menu.feature_y.addAction(act)
        self.menu.menubar.addMenu(self.menu.feature_y)
        self.menu.help_menu = self.menu.menubar.addMenu('Help')
        self.menu.help_menu.addAction(self.menu.help)
        # Statusbar
        self.menu.statusBar = self.window.statusBar()
        self.menu.statusBar.setSizeGripEnabled(False)
        # Indicate whether the user saved the current state
        self.menu.save_state = QtWidgets.QLabel("")
        self.menu.statusBar.addPermanentWidget(self.menu.save_state)
        # Toolbar
        self.menu.toolBar = QtWidgets.QToolBar('Tools')
        self.menu.toolBar.setStyleSheet('QToolBar {background: white}')
        self.window.addToolBar(QtCore.Qt.LeftToolBarArea, self.menu.toolBar)
        self.menu.toolBar.setFloatable(False)
        self.menu.toolBar.setMovable(True)
        self.menu.toolBar.addWidget(self.menu.toolbar_reset_zoom)
        self.menu.toolBar.addWidget(self.menu.toolbar_pan_zoom)
        self.menu.toolBar.addSeparator()
        self.menu.toolBar.addWidget(self.menu.add_cluster)
        self.menu.toolBar.addWidget(self.menu.edit_cluster)
        self.menu.toolBar.addWidget(self.menu.delete_cluster)
        self.menu.toolBar.addWidget(self.menu.merge_clusters)
        self.menu.toolBar.addSeparator()
        self.menu.toolBar.addWidget(self.menu.compute_CCG)
        self.menu.toolBar.addSeparator()
        self.menu.toolBar.addWidget(self.menu.compute_PC)
        self.menu.toolBar.addSeparator()
        self.menu.toolBar.addWidget(self.menu.compute_ISI)
        self.menu.toolBar.addWidget(self.menu.force_clean_ISI)

        # Make a vertical layout to push table at the top of its space
        vbox_table = QtWidgets.QVBoxLayout()
        vbox_table.addWidget(self.QTable)
        vbox_table.addStretch()
        # Add text edits to a 'form' widget
        form_bin = QtWidgets.QFormLayout()
        form_bin.addRow('Bin (ms)', self.textEdits.edit_ACG_bin)
        form_win = QtWidgets.QFormLayout()
        form_win.addRow('Window (ms)', self.textEdits.edit_ACG_lag)
        # Make horizontal layout with ACG buttons
        hbox_ACG = QtWidgets.QHBoxLayout()
        hbox_ACG.addStretch()
        hbox_ACG.addLayout(form_bin)
        hbox_ACG.addLayout(form_win)
        hbox_ACG.addStretch()
        # Make vertical layout with ACG plot and its control buttons at the top
        vbox_ACG = QtWidgets.QVBoxLayout()
        vbox_ACG.addLayout(hbox_ACG)
        vbox_ACG.addWidget(self.matplotlib_canvas.correlogram)
        # Enclose this layout in a widget
        widget_ACG = QtWidgets.QWidget()
        widget_ACG.setLayout(vbox_ACG)

        # Feature plot
        feature_layout = QtWidgets.QVBoxLayout()
        feature_layout.addWidget(self.matplotlib_canvas.feature)
        widget_feature = QtWidgets.QWidget()
        widget_feature.setLayout(feature_layout)

        # Waveform plot
        waveform_layout = QtWidgets.QHBoxLayout()
        self.fig['waveform'] = list()
        self.ax['waveform'] = list()
        self.matplotlib_canvas['waveform'] = list()
        self.matplotlib_toolbar['waveform'] = list()
        self.feature_time_range_slider = list()
        for ch in range(self.n_channels_this_shank):
            # Open figure with an axes in it
            self.fig['waveform'].append(Figure(facecolor='w'))
            if ch == 0:
                self.ax['waveform'].append(self.fig['waveform'][-1].add_subplot(1, 1, 1))
            else:
                self.ax['waveform'].append(self.fig['waveform'][-1].add_subplot(1, 1, 1, sharex=self.ax['waveform'][0], sharey=self.ax['waveform'][0]))
            # Copy figure to a FigureCanvas, and add a hidden toolbar (controlled
            # by callback functions)
            self.matplotlib_canvas['waveform'].append(OSSAFigureCanvas(self.fig['waveform'][-1], self.qFrame))
            # Create hidden toolbar for access to panning tool
            self.matplotlib_toolbar['waveform'].append(NavigationToolbar(self.matplotlib_canvas['waveform'][-1], self.qFrame))
            self.matplotlib_toolbar['waveform'][-1].hide()
            # Change resize policy of the graph to limit the minimum size
            self.matplotlib_canvas['waveform'][-1].setMinimumSize(self.OSSA.default['widget_minimum_size'], self.OSSA.default['widget_minimum_size'])
            self.matplotlib_canvas['waveform'][-1].updateGeometry()

            # Make a range slider to control the time range of feature analysis
            self.feature_time_range_slider.append(QRangeSlider(min_value=self.feature_time_range_ms[ch][0], max_value=self.feature_time_range_ms[ch][1], n_decimals=1))
            self.feature_time_range_slider[-1].setFixedHeight(20)

            # Make vertical layout to contain plot and slider
            channel_layout = QtWidgets.QVBoxLayout()
            channel_layout.addWidget(self.feature_time_range_slider[-1])
            channel_layout.addWidget(self.matplotlib_canvas.waveform[-1])
            waveform_layout.addLayout(channel_layout)

        # Assemble top layout
        widget_waveform = QtWidgets.QWidget()
        widget_waveform.setLayout(waveform_layout)
        # Add table and plots in a horizontal splitter for interactive resizing
        top_hor_splitter = make_QSplitter('hor')
        top_hor_splitter.addWidget(widget_ACG)
        top_hor_splitter.addWidget(widget_feature)
        top_hor_splitter.addWidget(widget_waveform)
        # Set the initial size
        top_hor_splitter.setStretchFactor(0, 1)
        top_hor_splitter.setStretchFactor(1, 2)
        top_hor_splitter.setStretchFactor(2, 1)
        # Combine all elements in the top row
        top_row = QtWidgets.QHBoxLayout()
        top_row.addLayout(vbox_table)
        top_row.addWidget(top_hor_splitter)
        widget_top_row = QtWidgets.QWidget()
        widget_top_row.setLayout(top_row)

        # Make a vertical layout with a splitter where to add the top row and the
        # stability plot at the bottom
        stability_layout = QtWidgets.QVBoxLayout()
        stability_layout.addWidget(self.matplotlib_canvas['stability'])
        widget_stability = QtWidgets.QWidget()
        widget_stability.setLayout(stability_layout)
        ver_splitter = make_QSplitter('ver')
        ver_splitter.addWidget(widget_top_row)
        ver_splitter.addWidget(widget_stability)
        # Enclose splitter in a widget
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(ver_splitter)

        # Assign layout to frame and focus on it
        self.qFrame.setLayout(layout)
        self.qFrame.setFocus()
        self.window.setCentralWidget(self.qFrame)

        # Connect callbacks
        # GUI resizing
        self.window.resized.connect(self.callback_on_resize)
        for plt_name in self._plot_names_apart_waveform:
            self.matplotlib_canvas[plt_name].resized.connect(self.callback_on_resize)
        [self.matplotlib_canvas['waveform'][ch].resized.connect(self.callback_on_resize) for ch in range(self.n_channels_this_shank)]
        # Point pickers
        self.callbacks.pick_spike[0] = self.fig.stability.canvas.mpl_connect('pick_event', partial(self.callback_show_clicked_waveform, from_plot='stability'))
        self.callbacks.pick_spike[1] = self.fig.feature.canvas.mpl_connect('pick_event', partial(self.callback_show_clicked_waveform, from_plot='feature'))
        # Menubar
        self.menu.help.triggered.connect(self.callback_show_help)
        self.menu.cancel.triggered.connect(partial(self.callback_quit, save=False))
        self.menu.save.triggered.connect(partial(self.callback_quit, save=True))
        self.menu.reload.triggered.connect(self.callback_reload)
        self.menu.undo.triggered.connect(partial(self.history_manager, action='undo'))
        self.menu.redo.triggered.connect(partial(self.history_manager, action='redo'))
        for f in features:
            for ii, plot_name in enumerate(['stability_y', 'feature_x', 'feature_y']):
                self.menu[f][ii].triggered.connect(partial(self.callback_change_plotted_features, plot_name=plot_name, feature=f))
        # Toolbar
        self.menu.toolbar_reset_zoom.clicked.connect(self.callback_reset_zoom)
        self.menu.toolbar_pan_zoom.clicked.connect(self.callback_pan_zoom_figure)
        self.menu.add_cluster.clicked.connect(self.callback_add_cluster)
        self.menu.edit_cluster.clicked.connect(self.callback_edit_cluster)
        self.menu.delete_cluster.clicked.connect(self.callback_delete_cluster)
        self.menu.merge_clusters.clicked.connect(self.callback_merge_clusters)
        self.menu.compute_ISI.clicked.connect(self.callback_compute_ISI_distributions)
        self.menu.force_clean_ISI.clicked.connect(self.callback_force_clean_ISI)
        self.menu.compute_CCG.clicked.connect(self.callback_compute_CCGs)
        self.menu.compute_PC.clicked.connect(self.callback_compute_PCs)
        # Text editors
        self.textEdits.edit_ACG_bin.returnPressed.connect(self.callback_recompute_ACG)
        self.textEdits.edit_ACG_lag.returnPressed.connect(self.callback_recompute_ACG)
        # Range slider
        [self.feature_time_range_slider[ch].startValueChanged.connect(partial(self.callback_feature_time_range_changed, 'start', ch)) for ch in range(self.n_channels_this_shank)]
        [self.feature_time_range_slider[ch].endValueChanged.connect(partial(self.callback_feature_time_range_changed, 'end', ch)) for ch in range(self.n_channels_this_shank)]
        # Table
        self.QTable.itemSelectionChanged.connect(self.callback_table_new_selection)
        self.table_assign_callbacks()
        # Make keyboard shortcuts
        self.keyboard.add_cluster = QtWidgets.QShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['SpikeSorter']['add_cluster']), self.window)
        self.keyboard.add_cluster.setAutoRepeat(False)
        self.keyboard.delete_cluster = QtWidgets.QShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['SpikeSorter']['delete_cluster']), self.window)
        self.keyboard.delete_cluster.setAutoRepeat(False)
        self.keyboard.edit_cluster = QtWidgets.QShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['SpikeSorter']['edit_cluster']), self.window)
        self.keyboard.edit_cluster.setAutoRepeat(False)
        self.keyboard.merge_clusters = QtWidgets.QShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['SpikeSorter']['merge_clusters']), self.window)
        self.keyboard.merge_clusters.setAutoRepeat(False)
        self.keyboard.pan_zoom_figure = QtWidgets.QShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['pan_zoom']), self.window)
        self.keyboard.pan_zoom_figure.setAutoRepeat(False)
        self.keyboard.reset_zoom = QtWidgets.QShortcut(QtGui.QKeySequence(self.OSSA.keyboard_shortcuts['reset_zoom']), self.window)
        self.keyboard.reset_zoom.setAutoRepeat(False)
        # Enable shortcuts
        self.keyboard.add_cluster.activated.connect(partial(self.callback_add_cluster, from_keyboard=True))
        self.keyboard.delete_cluster.activated.connect(self.callback_delete_cluster)
        self.keyboard.edit_cluster.activated.connect(partial(self.callback_edit_cluster, from_keyboard=True))
        self.keyboard.merge_clusters.activated.connect(self.callback_merge_clusters)
        self.keyboard.pan_zoom_figure.activated.connect(self.callback_pan_zoom_figure)
        self.keyboard.reset_zoom.activated.connect(self.callback_reset_zoom)

    def initialize_plots(self, include_selected_point=True):
        """Initialize the data in all the plots."""
        self._reset_correlogram_xy_data()
        self._update_ACG_times()

        # Loop through each cluster
        for idx in range(self.n_clusters):
            cluster_id = self.SUMMARY_TABLE.loc[idx, 'id']
            color = self.SUMMARY_TABLE.loc[idx, 'color']
            zorder = self.SUMMARY_TABLE.loc[idx, 'z_order']
            # Make a 'plot label' to store axes index
            label_cluster = '%i' % cluster_id

            # Get row in SUMMARY_TABLE
            row = self.summary_table_get_row_index(cluster_id)
            # Plot empty lines
            self.SUMMARY_TABLE.at[row, 'line_stability'] = self.ax.stability.plot(0, 0, alpha=1., marker='.', markersize=self.OSSA.default['scatter_size'], linestyle='None', markeredgecolor='None', markerfacecolor=color, markeredgewidth=2, picker=0, zorder=zorder, label=label_cluster)[0]
            self.SUMMARY_TABLE.at[row, 'line_feature'] = self.ax.feature.plot(0, 0, alpha=1., marker='.', markersize=self.OSSA.default['scatter_size'], linestyle='None', markeredgecolor='None', markerfacecolor=color, markeredgewidth=2, picker=0, zorder=zorder, label=label_cluster)[0]
            self.SUMMARY_TABLE.at[row, 'line_waveform'] = [self.ax.waveform[ch].plot(self.x_data['waveform'], self.y_data['waveform'][ch], linewidth=self.OSSA.default['line_width_thin'], color=color, zorder=zorder)[0] for ch in range(self.n_channels_this_shank)]
            self.SUMMARY_TABLE.at[row, 'line_correlogram'] = self.ax.correlogram.plot(self.x_data['correlogram'], self.y_data['correlogram'], linewidth=self.OSSA.default['line_width_thin'], color=color, zorder=zorder)[0]

        if include_selected_point:
            # Stability plot
            # Add a visible marker for the selected point
            self.last_selected_line.stability = self.ax.stability.plot(-1000, 0, linestyle='None', marker='o', markersize=self.OSSA.default['last_selected_point_size'], markerfacecolor='red', markeredgecolor='k', markeredgewidth=1.5, zorder=100, scalex=False, scaley=False)[0]
            # Plot a 0-line
            self.ax.stability.axhline(0, color='black', linewidth=1, zorder=99)
            # Add text box showing last selected point coordinates
            self.text.stability = self.ax.stability.annotate("", xy=(.01, .99), xytext=(.01, .99), xycoords='axes fraction', ha='left', va='top', clip_on=True, bbox=self.OSSA.default['SpikeSorter']['stability_plot_textbox_props'], zorder=100)

            # Feature plot
            # Add a visible marker for the selected point
            self.last_selected_line.feature = self.ax.feature.plot(-1000, 0, linestyle='None', marker='o', markersize=self.OSSA.default['last_selected_point_size'], markerfacecolor='red', markeredgecolor='k', markeredgewidth=1.5, zorder=100, scalex=False, scaley=False)[0]
            self.ax.feature.axhline(0, color='black', linewidth=1, zorder=99)
            self.ax.feature.axvline(0, color='black', linewidth=1, zorder=99)

            # Waveform plot
            self.last_selected_line.waveform = [self.ax.waveform[ch].plot(self.x_data['waveform'], self.y_data['waveform'][ch], color='k', linewidth=self.OSSA.default['line_width_thick'], zorder=1000)[0] for ch in range(self.n_channels_this_shank)]
            # Plot zero-lines
            [self.ax.waveform[ch].axhline(0, color=(.7, .7, .7), linewidth=1, zorder=1) for ch in range(self.n_channels_this_shank)]
            [self.ax.waveform[ch].axvline(0, color=(.7, .7, .7), linewidth=1, zorder=1) for ch in range(self.n_channels_this_shank)]

            # Correlogram plot
            self.ax.correlogram.axhline(0, color=(.7, .7, .7), linewidth=1, zorder=1)

        # Fix axes appearance of stability plot
        stability_plot_xticks_samples = np.concatenate(([0], self.CS.INFO['segments'][:, 1]))
        stability_plot_xticks_min = np.round(stability_plot_xticks_samples / self.CS.INFO['sampling_frequency'] / 60.)
        stability_plot_xticklabels = [str('%i' % i) for i in stability_plot_xticks_min]
        self.ax.stability.set_xticks(stability_plot_xticks_samples)
        self.ax.stability.set_xticklabels(list())
        self.ax.stability.set_xticklabels(stability_plot_xticklabels)
        self.ax.stability.set_xlabel('Time (min)')
        ylabel = translate_feature_to_axis_label(self.OSSA.plotted_features['SpikeSorter']['y']['stability'])
        self.ax.stability.set_ylabel(ylabel)

        # Fix axes appearance of feature plot
        self.ax.feature.set_xlabel(translate_feature_to_axis_label(self.OSSA.plotted_features['SpikeSorter']['x']['feature']))
        self.ax.feature.set_ylabel(translate_feature_to_axis_label(self.OSSA.plotted_features['SpikeSorter']['y']['feature']))

        # Fix axes appearance of waveform plot
        for ch in range(self.n_channels_this_shank):
            # Reset ticks and labels from all plots
            self.ax.waveform[ch].set_xticks(list())
            self.ax.waveform[ch].set_yticks(list())
            # Find x-lims and apply them
            self.ax.waveform[ch].set_xlim(self.x_data['waveform'].min(), self.x_data['waveform'].max())
            # Make better x-ticks
            waveform_time = np.linspace(-self.CS.waveform_before_peak_ms, self.CS.waveform_after_peak_ms, self.CS.spike_waveform_duration)
            waveform_plot_xticklabels = np.arange(-self.CS.waveform_before_peak_ms, self.CS.waveform_after_peak_ms + .001, 0.5)
            xticks = np.array([np.abs(t - waveform_time).argmin() for t in waveform_plot_xticklabels])
            self.ax.waveform[ch].set_xticks(xticks)
            self.ax.waveform[ch].set_xticklabels(waveform_plot_xticklabels)
            self.ax.waveform[ch].set_xlabel('Time (ms)')

        # Fix axes appearance of correlogram plot
        self.ax.correlogram.set_xlabel('Lag (ms)')
        self.ax.correlogram.set_yticks(list())
        self.correlogram_make_xlims()

        # Plot data
        self.reset_figures()

    def _is_first_run(self):
        """Determine whether this is the first time user runs spike sorting on
        these channels."""
        row = list(self.OSSA.OSSA_results['channel']).index(self.CS.shank_selection_names)
        already_processed_once = self.OSSA.OSSA_results.loc[row, 'SpikeSorter']
        return not already_processed_once

    ############################################################################
    # User interaction with GUI window
    ############################################################################
    def show_GUI(self):
        """Short-hand function to show GUI"""
        self.window.showMaximized()
        self.window.raise_()
        # Change flag
        self.OSSA.currently_showing = 'SpikeSorter'

    def on_close(self):
        # Close children windows left open by the user (CCG, ISI)
        open_figs = plt.get_fignums()
        for fig in open_figs:
            plt.close(fig)
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
        msg = ""
        last_state = list()

        if action == 'initialize':
            initialize_field(self, 'HISTORY')
            initialize_field(self.HISTORY, ['undo', 'redo'], 'list', shape=self.OSSA.analysis['history_size'])
            for history_item in range(self.OSSA.analysis['history_size']):
                self.HISTORY.undo[history_item] = list(np.empty((3, ), dtype=object))
                self.HISTORY.redo[history_item] = list(np.empty((3,), dtype=object))
            self.HISTORY.first_state = list(np.empty((3, ), dtype=object))
            self.HISTORY.current_state = list(np.empty((3,), dtype=object))
            return

        elif action == 'reset':
            msg = 'INFO: Resetting history manager to last save'
            # Reset cluster ids
            self.DATA['cluster_id_manual'] = self.HISTORY.first_state[0]
            # Remove all graphical elements and re-initialize SUMMARY_TABLE
            for row in range(self.SUMMARY_TABLE.shape[0]):
                for plt_name in self._plot_names:
                    self.ax[plt_name].lines.remove(self.SUMMARY_TABLE.loc[row, 'line_%s' % plt_name])

            # Reset SUMMARY TABLE
            self.reset_summary_table(from_disk=True)
            # Add containers for line handles
            for plt_name in self._plot_names:
                self.SUMMARY_TABLE['line_%s' % plt_name] = None
            self.n_clusters = self.SUMMARY_TABLE.shape[0]
            # Initialize history
            initialize_field(self.HISTORY, ['undo', 'redo'], 'list', self.OSSA.analysis['history_size'])
            for history_item in range(self.OSSA.analysis['history_size']):
                self.HISTORY.undo[history_item] = list(np.empty((3, ), dtype=object))
                self.HISTORY.redo[history_item] = list(np.empty((3,), dtype=object))
            self.HISTORY.current_state = list(np.empty((3,), dtype=object))
            # Compute correlograms
            self._initialize_correlograms()
            # Initialize all plots
            self.initialize_plots(include_selected_point=False)

        elif action == 'do':
            # Add latest state to the 'current' item of the history.
            if self.HISTORY.current_state[0] is not None:
                # Add latest state to the 'undo' stack.
                self.HISTORY.undo.pop(0)
                self.HISTORY.undo.append(self.HISTORY.current_state[:])
            # Re-initialize the 'redo' stack if not empty
            if self.HISTORY.redo[-1][0] is not None:
                for history_item in range(self.OSSA.analysis['history_size']):
                    self.HISTORY.redo[history_item] = list(np.empty((3,), dtype=object))

        elif action == 'undo':
            if self.HISTORY.undo[-1][0] is not None:  # State can be undone
                msg = 'INFO: Undone last action'
                # Push current state to the 'redo' stack
                self.HISTORY.redo.pop(0)
                self.HISTORY.redo.append(self.HISTORY.current_state[:])
                # Pop the last element of the 'undo' stack
                last_state = self.HISTORY.undo.pop(-1)
                # Copy cluster ids and correlograms
                self.DATA['cluster_id_manual'] = last_state[0]
                self.correlograms = last_state[1]
                # Add an empty element at the beginning of the 'undo' stack
                self.HISTORY.undo.insert(0, list(np.empty((3,), dtype=object)))
            else:
                return

        elif action == 'redo':
            if self.HISTORY.redo[-1][0] is not None:  # State can be redone
                msg = 'INFO: Redone last action'
                # Push current state to the 'undo' stack
                self.HISTORY.undo.pop(0)
                self.HISTORY.undo.append(self.HISTORY.current_state)
                # Move the last element of the 'redo' stack to the current state
                last_state =self.HISTORY.redo.pop(-1)
                # Copy cluster ids and correlograms
                self.DATA['cluster_id_manual'] = last_state[0]
                self.correlograms = last_state[1]
                # Add an empty element at the beginning of the 'redo' stack
                self.HISTORY.redo.insert(0, list(np.empty((3,), dtype=object)))
            else:
                return

        # Operations to do that are common to 'undo' and 'redo'
        if action in ['undo', 'redo']:
            # Store the current axes limits
            prev_ax_lims = dict()
            for plt_name in self._plot_names:
                prev_ax_lims[plt_name] = [self.ax[plt_name].get_xlim(), self.ax[plt_name].get_ylim()]
            # Reset all graphs
            for row in range(self.SUMMARY_TABLE.shape[0]):
                for plt_name in self._plot_names:
                    self.ax[plt_name].lines.remove(self.SUMMARY_TABLE.loc[row, 'line_%s' % plt_name])
            # Restore data from backup
            self.SUMMARY_TABLE = pd.DataFrame(columns=last_state[2].columns)
            self.SUMMARY_TABLE['id'] = last_state[2]['id']
            self.SUMMARY_TABLE['n_spikes'] = last_state[2]['n_spikes']
            self.SUMMARY_TABLE['FR'] = last_state[2]['FR']
            self.SUMMARY_TABLE['cluster_type'] = last_state[2]['cluster_type']
            self.SUMMARY_TABLE['color'] = last_state[2]['color']
            # Re-create z-order
            self.summary_table_update_zorder()
            # Update CCG and SUMMARY TABLE indices
            self._update_cluster_count_and_row_index()
            # Reset all plots
            self.initialize_plots(include_selected_point=False)
            # Remove clusters that do not exist anymore from the 'currently
            # selected' list
            self.currently_selected_clusters = np.intersect1d(self.SUMMARY_TABLE['id'].unique(), self.currently_selected_clusters, assume_unique=True)
            if self.currently_selected_clusters.shape[0] == 0:
                self.currently_selected_clusters = np.array([0], dtype=int)
            # Re-select clusters on table
            self._table_focus_on_cluster(self.currently_selected_clusters)

            # Restore zoom
            for plt_name in self._plot_names:
                self.ax[plt_name].set_xlim(prev_ax_lims[plt_name][0])
                self.ax[plt_name].set_ylim(prev_ax_lims[plt_name][1])
                self.ax_lims[plt_name] = [self.ax[plt_name].get_xlim(), self.ax[plt_name].get_ylim()]
                self.ax[plt_name].figure.canvas.draw()

        # Update table widget
        self.QTable.update_table_data(self.SUMMARY_TABLE.copy())
        self.table_assign_callbacks()
        # Log outcome
        LOGGER.info(msg)

        # Reset indices in tables
        self._update_cluster_count_and_row_index()
        # Update current state before it gets modified externally
        self.HISTORY.current_state = [backup(self.DATA['cluster_id_manual'].values), backup(self.correlograms), backup(self.SUMMARY_TABLE)]

    def table_assign_callbacks(self):
        """Assign the color picking callback to the table, by passing the
        function and the rows where the button is located."""
        for row, cluster_id in enumerate(self.SUMMARY_TABLE['id'].values):
            if cluster_id != 0:
                # self.window.connect(self.QTable.color_picker[row], QtCore.SIGNAL('clicked()'), partial(self.callback_table_pick_a_color, cluster_id=cluster_id))
                self.QTable.color_picker[row].clicked.connect(partial(self.callback_table_pick_a_color, cluster_id=cluster_id))
                self.QTable.cluster_type_picker[row].currentIndexChanged.connect(partial(self.callback_table_change_cluster_type, cluster_id=cluster_id))

    def toggle_button_pan_zoom(self, toolbar_state):
        """This method toggles the button to pan and zoom"""
        # Get the current state
        current_state = self.currently_panning
        # Continue only if state changed
        if current_state != toolbar_state:
            # Toggle state variable
            self.currently_panning = toolbar_state
            # Toggle panning on all axes
            for plt_name in self._plot_names_apart_waveform:
                self.matplotlib_toolbar[plt_name].pan()
            [self.matplotlib_toolbar['waveform'][ch].pan() for ch in range(self.n_channels_this_shank)]
        # Toggle toolbar button
        self.menu.toolbar_pan_zoom.setChecked(toolbar_state)

    def set_all_buttons(self, to, except_button):
        """Quickly toggles all buttons except the 'add' or 'edit' button, as set
         by <add_or_edit>."""
        if to:
            self.menu.menubar.setEnabled(to)
            for txt_name in self._button_text_edit:
                self.textEdits[txt_name].setEnabled(to)
            self.update_button_state_from_table_selection(self.currently_selected_clusters)
        else:
            # Toggle all buttons in the GUI
            for btn_name in self._button_names_plot:
                self.menu[btn_name].setEnabled(to)
            for txt_name in self._button_text_edit:
                self.textEdits[txt_name].setEnabled(to)
            self.menu.menubar.setEnabled(to)

            # 'add' and 'edit' buttons are toggled by callbacks
            if except_button == 'add':
                self.menu.add_cluster.setEnabled(not to)
                self.menu.edit_cluster.setEnabled(to)
                self.menu.edit_cluster.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(self.OSSA.default['SpikeSorter']['button_color_neutral'], dtype=float)*255))))
            elif except_button == 'edit':
                self.menu.edit_cluster.setEnabled(not to)
                self.menu.add_cluster.setEnabled(to)
                self.menu.add_cluster.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(self.OSSA.default['SpikeSorter']['button_color_neutral'], dtype=float)*255))))

    def _update_ACG_times(self):
        """Update text editor QWidgets that show current values of bin size and
        lag in GUI"""
        self.textEdits.edit_ACG_bin.setText(str('%i' % self.CCG_bin_size))
        self.textEdits.edit_ACG_lag.setText(str('%i' % self.CCG_max_lag))

    def reset_summary_table(self, from_disk, store_to_disk=False):
        """Reset the SUMMARY_TABLE according to data in memory or on disk."""
        if from_disk:
            self.SUMMARY_TABLE = read_sorting_summary_table(self.OSSA.spikes_hdf5_filename, self.CS.shank_selection_names)

        else:
            self.SUMMARY_TABLE = pd.DataFrame(columns=['shank', 'id', 'n_spikes', 'FR', 'color', 'cluster_type', 'z_order'])
            # Get unique cluster ids from the data (and make sure 0 is included)
            all_cluster_ids = np.unique(np.hstack(([0], self.DATA['cluster_id_manual'])))
            self.n_clusters = len(all_cluster_ids)

            for idx, cluster_id in enumerate(all_cluster_ids):
                cluster_spikes = self.get_spike_train(cluster_id)
                n_spikes = len(cluster_spikes)
                if n_spikes == 0 and cluster_id != 0:
                    continue
                # Add cluster type (MU / SU / noise) to summary table
                if cluster_id == 0:
                    cluster_type = 'n'
                    cluster_color = self.OSSA.default['SpikeSorter']['color_unsorted_spikes']
                else:
                    cluster_type = 'm'
                    cluster_color = next(self.CLUSTER_COLORS)
                table_index = self.SUMMARY_TABLE.shape[0]
                self.SUMMARY_TABLE.at[table_index, 'shank'] = self.CS.shank_selection_names
                self.SUMMARY_TABLE.at[table_index, 'id'] = cluster_id
                self.SUMMARY_TABLE.at[table_index, 'n_spikes'] = n_spikes
                self.SUMMARY_TABLE.at[table_index, 'FR'] = n_spikes / self.CS.INFO['recording_duration']
                self.SUMMARY_TABLE.at[table_index, 'color'] = cluster_color
                self.SUMMARY_TABLE.at[table_index, 'cluster_type'] = cluster_type

        # Make sure that cluster 0 is the first one
        self.SUMMARY_TABLE.sort_values(by=['shank', 'id'], inplace=True)
        self.SUMMARY_TABLE.reset_index(drop=True, inplace=True)
        # Update number of clusters and line z-order
        self.n_clusters = self.SUMMARY_TABLE.shape[0]
        self.summary_table_update_zorder()

        # Add containers for line handles
        for plt_name in self._plot_names:
            self.SUMMARY_TABLE['line_%s' % plt_name] = None

        # Write summary table to disk
        if store_to_disk:
            write_sorting_summary_table(self.CS.spikes_hdf5_filename, self.SUMMARY_TABLE)


    ############################################################################
    # User interaction with figures
    ############################################################################
    def reset_figures(self):
        """Reset all figures."""
        # Restore plots status
        self.update_all_plots()
        # Selected spike
        self.last_selected_line.stability.set_ydata(None)
        self.last_selected_line.feature.set_ydata(None)
        [self.last_selected_line.waveform[ch].set_ydata(self.y_data['waveform'][ch].shape[0]) for ch in range(self.n_channels_this_shank)]
        # Store the current x- and y-limits
        for plt_name in self._plot_names_apart_waveform:
            self.ax_lims[plt_name] = [self.ax[plt_name].get_xlim(), self.ax[plt_name].get_ylim()]
        self.ax_lims['waveform'] = [[self.ax['waveform'][ch].get_xlim(), self.ax['waveform'][ch].get_ylim()] for ch in range(self.n_channels_this_shank)]

    def _update_plot(self, cluster_list, plot_name, update_data=True, update_limits=False):
        # Initialize variables
        x = None
        y = None

        # Make sure cluster_list is a list
        if not isinstance(cluster_list, (list, np.ndarray)):
            cluster_list = [cluster_list]
        # Hide all clusters
        if update_data:
            for row in range(self.SUMMARY_TABLE.shape[0]):
                if plot_name == 'waveform':
                    [self.SUMMARY_TABLE.loc[row, 'line_%s' % plot_name][ch].set_visible(False) for ch in range(self.n_channels_this_shank)]
                else:
                    self.SUMMARY_TABLE.loc[row, 'line_%s' % plot_name].set_visible(False)
                if plot_name in ['stability', 'feature']:
                    self.SUMMARY_TABLE.loc[row, 'line_%s' % plot_name].set_picker(None)
        # Set new data and visibility
        xlims = np.zeros((self.n_clusters, 2), dtype=float)
        ylims = np.zeros_like(xlims)
        for idx, cluster_id in enumerate(cluster_list):
            table_row = self.summary_table_get_row_index(cluster_id)
            zorder = self.SUMMARY_TABLE.loc[table_row, 'z_order']
            # Get data
            if plot_name == 'waveform':
                cluster_spikes = self.get_spike_waveforms(cluster_id)
            else:
                cluster_spikes = self.get_spike_train(cluster_id)
            # Decide what to do based on the number of spikes and user inputs
            n_spikes = cluster_spikes.shape[0]
            if n_spikes > 0 or update_data:
                # Set data
                if plot_name == 'stability':
                    x = self.DATA['timestamp'][cluster_spikes]
                    y = self.DATA[self.OSSA.plotted_features['SpikeSorter']['y']['stability']][cluster_spikes]
                elif plot_name == 'feature':
                    x = self.DATA[self.OSSA.plotted_features['SpikeSorter']['x']['feature']][cluster_spikes]
                    y = self.DATA[self.OSSA.plotted_features['SpikeSorter']['y']['feature']][cluster_spikes]
                elif plot_name == 'waveform':
                    x = self.x_data['waveform']
                    if n_spikes == 0:
                        y = [np.zeros_like(x) for _ in range(self.n_channels_this_shank)]
                    else:
                        y = np.mean(cluster_spikes, axis=0).transpose().tolist()
                elif plot_name == 'correlogram':
                    self._reset_correlogram_xy_data()
                    y = self.correlograms_get_data(cluster_id, cluster_id)
                    x, y = self.correlogram_make_xy_outline(self.x_data['correlogram'], y)
                # Assign data to graphical elements
                if update_data:
                    # Update data, z-order and visibility
                    if plot_name == 'waveform':
                        [self.SUMMARY_TABLE.loc[table_row, 'line_waveform'][ch].set_data(x, y[ch]) for ch in range(self.n_channels_this_shank)]
                        [self.SUMMARY_TABLE.loc[table_row, 'line_waveform'][ch].set_zorder(zorder) for ch in range(self.n_channels_this_shank)]
                        [self.SUMMARY_TABLE.loc[table_row, 'line_waveform'][ch].set_visible(True) for ch in range(self.n_channels_this_shank)]
                    else:
                        self.SUMMARY_TABLE.loc[table_row, 'line_%s' % plot_name].set_data(x, y)
                        self.SUMMARY_TABLE.loc[table_row, 'line_%s' % plot_name].set_zorder(zorder)
                        self.SUMMARY_TABLE.loc[table_row, 'line_%s' % plot_name].set_visible(True)
                    # Re-enable the picker
                    if plot_name in ['stability', 'feature']:
                        self.SUMMARY_TABLE.loc[table_row, 'line_%s' % plot_name].set_picker(self.waveform_picker_tolerance)
                # Update data limits
                if n_spikes > 0:
                    xlims[idx, :] = compute_array_range(x, padding=False)
                    ylims[idx, :] = compute_array_range(y, padding=False)
        # Set new limits
        if update_limits:
            self.set_plot_lims(xlims, ylims, plot_name)
        # Update canvas
        if plot_name == 'waveform':
            [self.ax['waveform'][ch].figure.canvas.draw() for ch in range(self.n_channels_this_shank)]
        else:
            self.ax[plot_name].figure.canvas.draw()

    def update_all_plots(self):
        """Recompute data and limits of all plots."""
        for plt_name in self._plot_names:
            self._update_plot(self.SUMMARY_TABLE['id'].values, plt_name, update_limits=True)

    def set_plot_lims(self, x, y, on_plot):
        """Set the limits for a plot."""
        # Remove invalid numbers from lists
        x = x.ravel()[np.isfinite(x.ravel())]
        y = y.ravel()[np.isfinite(y.ravel())]
        # Check whether limits changed, otherwise don't reset the zoom
        changed_limits = False
        for dim in [0, 1]:
            # Get the data for this dimension
            if dim == 0:
                this_data = x
            else:
                this_data = y
            if len(this_data) != 0:
                # Don't add padding to x-axis of correlogram and waveform plot
                if (on_plot == 'correlogram' and dim==0) or (on_plot == 'waveform' and dim==0):
                    add_padding = False
                else:
                    add_padding = True
                # Take extrema
                lims = compute_array_range(this_data, padding=add_padding)
                # Save limits and mark that the limits have changed
                if lims[1] - lims[0] == 0:
                    lims[1] += 1
                self.ax_lims[on_plot][dim] = lims
                changed_limits = True
        # Reset zoom if limits have changed
        if changed_limits:
            if on_plot == 'waveform':
                [self.ax[on_plot][ch].set_xlim(self.ax_lims[on_plot][0]) for ch in range(self.n_channels_this_shank)]
                [self.ax[on_plot][ch].set_ylim(self.ax_lims[on_plot][1]) for ch in range(self.n_channels_this_shank)]
            else:
                self.ax[on_plot].set_xlim(self.ax_lims[on_plot][0])
                self.ax[on_plot].set_ylim(self.ax_lims[on_plot][1])

    def update_figures(self, update_limits):
        """Update data in all figures to show only the selected clusters."""
        n_selected_clusters = len(self.currently_selected_clusters)
        if n_selected_clusters > 0:
            for plt_name in self._plot_names:
                self._update_plot(self.currently_selected_clusters, plt_name, update_limits=update_limits)
        else:  # no clusters are currently selected
            for idx in range(self.n_clusters):
                for plt_name in self._plot_names_apart_waveform:
                    self.SUMMARY_TABLE.loc[idx, 'line_%s' % plt_name].set_visible(False)
                    # Remove picker function from stability and feature plots
                    self.SUMMARY_TABLE.loc[idx, 'line_%s' % plt_name].set_picker(None)
                    self.SUMMARY_TABLE.loc[idx, 'line_%s' % plt_name].set_picker(None)
                [self.SUMMARY_TABLE.loc[idx, 'line_waveform'][ch].set_visible(False) for ch in range(self.n_channels_this_shank)]
                [self.SUMMARY_TABLE.loc[idx, 'line_waveform'][ch].set_picker(None) for ch in range(self.n_channels_this_shank)]
                [self.SUMMARY_TABLE.loc[idx, 'line_waveform'][ch].set_picker(None) for ch in range(self.n_channels_this_shank)]
        # Update canvas
        for plt_name in self._plot_names_apart_waveform:
            self.ax[plt_name].figure.canvas.draw()
        [self.ax['waveform'][ch].figure.canvas.draw() for ch in range(self.n_channels_this_shank)]

    def show_only_cluster(self, cluster_id, update_limits=True):
        """Method to highlight only a particular cluster."""
        if hasattr(cluster_id, '__iter__'):
            self.currently_selected_clusters = np.array(cluster_id)
        else:
            self.currently_selected_clusters = np.array([cluster_id])
        self.update_figures(update_limits=update_limits)

    def update_button_state_from_table_selection(self, rows):
        """Respond interactively to table selection to update the GUI."""
        n_rows = len(np.unique(rows))
        # Some buttons can only be active if only 1, at least 1 or at least 2
        # clusters are selected.
        if n_rows == 0:  # no clusters selected
            state_button_at_least_2_clusters = False
            state_button_at_least_1_cluster = False
            state_button_only_1_cluster = False
        elif n_rows == 1:  # only 1 cluster selected
            state_button_at_least_2_clusters = False
            state_button_at_least_1_cluster = True
            state_button_only_1_cluster = True
        else:  # >=1 cluster selected
            state_button_at_least_2_clusters = True
            state_button_at_least_1_cluster = True
            state_button_only_1_cluster = False
        # Toggle button state
        self.menu.merge_clusters.setEnabled(state_button_at_least_2_clusters)
        self.menu.compute_ISI.setEnabled(state_button_at_least_1_cluster)
        self.menu.force_clean_ISI.setEnabled(state_button_only_1_cluster)
        self.menu.compute_CCG.setEnabled(state_button_at_least_2_clusters)
        self.menu.compute_PC.setEnabled(state_button_at_least_1_cluster)
        self.menu.add_cluster.setEnabled(state_button_at_least_1_cluster)
        self.menu.edit_cluster.setEnabled(state_button_at_least_1_cluster)
        self.menu.delete_cluster.setEnabled(state_button_at_least_1_cluster)
        if state_button_at_least_1_cluster:
            self.menu.add_cluster.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(self.OSSA.default['SpikeSorter']['button_color_add'], dtype=float)*255))))
            self.menu.edit_cluster.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(self.OSSA.default['SpikeSorter']['button_color_delete'], dtype=float)*255))))
        else:
            self.menu.add_cluster.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(self.OSSA.default['SpikeSorter']['button_color_neutral'], dtype=float)*255))))
            self.menu.edit_cluster.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(self.OSSA.default['SpikeSorter']['button_color_neutral'], dtype=float)*255))))

        # If only 1 cluster is selected, adapt button state even more precisely
        if state_button_only_1_cluster:
            # If cluster 0 is selected, disable the force_clean_ISI button
            if self.currently_selected_clusters[0] == 0:
                self.menu.force_clean_ISI.setEnabled(False)

            # Get number of spikes in cluster
            row = self.summary_table_get_row_index(self.currently_selected_clusters[0])
            n_spikes = self.SUMMARY_TABLE.loc[row, 'n_spikes']

            # ISI histograms and PCs need at least 2 spikes
            if n_spikes >= 2:
                self.menu.compute_ISI.setEnabled(True)
                self.menu.force_clean_ISI.setEnabled(True)
                self.menu.compute_PC.setEnabled(True)
            else:
                self.menu.compute_ISI.setEnabled(False)
                self.menu.force_clean_ISI.setEnabled(False)
                self.menu.compute_PC.setEnabled(False)

    def _add_line_for_new_cluster(self, cluster_id, zorder):
        """Add a graphical element for a new cluster"""
        # Make a 'plot label' to store axes index
        label_channel = '%i' % cluster_id
        row = self.summary_table_get_row_index(cluster_id)
        color = self.SUMMARY_TABLE.loc[row, 'color']
        self.SUMMARY_TABLE.loc[row, 'line_stability'] = self.ax.stability.plot(0, 0, alpha=1., marker='.', markersize=self.OSSA.default['scatter_size'], linestyle='None', markeredgecolor='None', markerfacecolor=color, markeredgewidth=2, picker=self.waveform_picker_tolerance, zorder=zorder, label=label_channel)[0]
        self.SUMMARY_TABLE.loc[row, 'line_feature'] = self.ax.feature.plot(0, 0, alpha=1., marker='.', markersize=self.OSSA.default['scatter_size'], linestyle='None', markeredgecolor='None', markerfacecolor=color, markeredgewidth=2, picker=self.waveform_picker_tolerance, zorder=zorder, label=label_channel)[0]
        self.SUMMARY_TABLE.loc[row, 'line_waveform'] = [self.ax.waveform[ch].plot(self.x_data['waveform'], self.y_data['waveform'][ch], linewidth=self.OSSA.default['line_width_thin'], alpha=1., color=color, zorder=zorder)[0] for ch in range(self.n_channels_this_shank)]
        self.SUMMARY_TABLE.loc[row, 'line_correlogram'] = self.ax.correlogram.plot(self.x_data['correlogram'], self.y_data['correlogram'], linewidth=self.OSSA.default['line_width_thin'], alpha=1., color=color, zorder=zorder)[0]

    def _table_focus_on_cluster(self, cluster_list):
        """Force the table to focus on a specific cluster."""
        # Get row indices to highlight
        if not isinstance(cluster_list, (list, np.ndarray)):
            cluster_list = list([cluster_list])
        highlight_rows = list()
        for cluster_id in cluster_list:
            highlight_rows.append(self.QTable.get_row_containing(cluster_id))
        # Disable table interactivity
        self.QTable.disable_interaction()
        self.QTable.clearSelection()
        # Loop through rows
        for row in highlight_rows:
            self.QTable.selectRow(row)

        # Enable table interactivity
        self.QTable.enable_interaction()
        # Call function to update plots based on selected rows in table
        self.callback_table_new_selection(update_limits=False)

    def update_last_selected_point(self, x_stability, y_stability, x_feature, y_feature, color):
        """Update the point marking the last selected point"""
        self.last_selected_line.stability.set_xdata(x_stability)
        self.last_selected_line.stability.set_ydata(y_stability)
        self.last_selected_line.stability.set_markerfacecolor(color)
        self.last_selected_line.feature.set_xdata(x_feature)
        self.last_selected_line.feature.set_ydata(y_feature)
        self.last_selected_line.feature.set_markerfacecolor(color)

    def update_plotted_data_of_cluster_id(self, cluster_id):
        """Update the data of a specific cluster."""
        self.summary_table_update_zorder()
        for plot_name in self._plot_names:
            self._update_plot(cluster_id, plot_name, update_limits=False)

    def start_polygon_selector(self, add_or_edit, from_keyboard):
        """Routine to pass currently visible points to the polygon selector, and
        activate user interaction with it."""
        # Disable table and the pan/zoom tool
        self.QTable.disable_interaction()
        self.toggle_button_pan_zoom(False)
        # Disable all buttons except the current one to cancel / accept selection
        self.set_all_buttons(to=False, except_button=add_or_edit)
        # Mark the current state
        self.currently_selecting_spikes = True
        # It can happen that the lines that are visible are not currently
        # selected in the table. Then, check which clusters are currently visible
        # assuming that the user is referring to what he/she sees plotted and not
        # what is currently selected in the table!
        self.currently_selected_clusters = list()
        for idx in range(self.n_clusters):
            if self.SUMMARY_TABLE.loc[idx, 'line_stability'].get_visible():
                cluster_id = self.SUMMARY_TABLE.loc[idx, 'id']
                self.currently_selected_clusters.append(cluster_id)
        self.currently_selected_clusters = np.array(self.currently_selected_clusters, dtype=int)
        self.currently_selected_clusters.sort()
        # Get data
        x_stability = np.empty((0, ), dtype=float)
        y_stability = np.empty_like(x_stability)
        x_feature = np.empty_like(x_stability)
        y_feature = np.empty_like(x_stability)
        spike_id = np.empty_like(x_stability, dtype=int)
        for cluster_id in self.currently_selected_clusters:
            cluster_line = self.summary_table_get_row_index(cluster_id)
            x_s = self.SUMMARY_TABLE.loc[cluster_line, 'line_stability'].get_xdata()
            y_s = self.SUMMARY_TABLE.loc[cluster_line, 'line_stability'].get_ydata()
            x_f = self.SUMMARY_TABLE.loc[cluster_line, 'line_feature'].get_xdata()
            y_f = self.SUMMARY_TABLE.loc[cluster_line, 'line_feature'].get_ydata()
            ind = self.get_spike_train(cluster_id)

            x_stability = np.concatenate((x_stability, x_s))
            y_stability = np.concatenate((y_stability, y_s))
            x_feature = np.concatenate((x_feature, x_f))
            y_feature = np.concatenate((y_feature, y_f))
            spike_id = np.concatenate((spike_id, ind))
        # Activate polygon selector
        if add_or_edit == 'add':
            color = self.OSSA.default['SpikeSorter']['selector_color_add']
        else:
            color = self.OSSA.default['SpikeSorter']['selector_color_discard']
        self.callbacks.polygon_selectors.stability = SelectFromCollection(self.ax.stability, x_stability, y_stability, spike_id, color, partial(self.validate_polygon_selection, add_or_edit=add_or_edit, from_keyboard=from_keyboard))
        self.callbacks.polygon_selectors.feature = SelectFromCollection(self.ax.feature, x_feature, y_feature, spike_id, color, partial(self.validate_polygon_selection, add_or_edit=add_or_edit, from_keyboard=from_keyboard))

    def validate_polygon_selection(self, add_or_edit, from_keyboard=False):
        """Callback to accept the spike selection and update data and plots."""
        # Continue only if the exit code from the polygon selectors was not equal to 0
        if not np.any(np.array([self.callbacks.polygon_selectors.stability.exit_code, self.callbacks.polygon_selectors.feature.exit_code]) == 0):
            valid_selection = True
            # If polygon is not closed, clicking doesn't have any effect
            if not any((self.callbacks.polygon_selectors.stability.polygon_completed, self.callbacks.polygon_selectors.feature.polygon_completed)):
                valid_selection = False
            n_selected_spikes, selected_spikes = self.get_indices_of_points_in_selector()
            if n_selected_spikes == 0:
                valid_selection = False
            if valid_selection:
                if add_or_edit == 'add':
                    # Get the id of the new cluster, print that to the console
                    recipient_cluster_id = self.make_new_cluster_id()
                    msg = 'Making cluster %i from %i spikes' % (recipient_cluster_id, n_selected_spikes)
                else:
                    recipient_cluster_id = 0
                    msg = 'Unclustering %i spikes' % n_selected_spikes
                # Show message in log, console and statusbar
                self.show_in_console_and_statusbar(msg)
                # Move spikes to new cluster
                from_clusters = list(np.unique(self.DATA.loc[selected_spikes, 'cluster_id_manual'].values))
                self.move_spikes_to_cluster(selected_spikes, to_cluster_id=recipient_cluster_id)
                # Loop through all modified clusters to check which ones have to
                # be deleted or updated
                visible_clusters = list(self.currently_selected_clusters)
                clusters_to_delete = list()
                for cluster_id in from_clusters:
                    # Update summary table
                    self.summary_table_update_row(cluster_id)
                    # If no spikes are left in this old cluster id, delete it
                    if self.SUMMARY_TABLE.loc[self.summary_table_get_row_index(cluster_id), 'n_spikes'] == 0:
                        clusters_to_delete.append(cluster_id)
                    else:  # Otherwise, update the ACG
                        self.correlograms_update_ACG_data(cluster_id)
                        self.update_plotted_data_of_cluster_id(cluster_id)
                # Remove empty clusters
                for cluster_id in clusters_to_delete:
                    if cluster_id != 0:
                        # Delete cluster from the data
                        self._remove_cluster(cluster_id)
                        # Remove it from the list of current clusters
                        visible_clusters.remove(cluster_id)
                # Update info and ACG of new cluster
                if add_or_edit == 'add':
                    self.summary_table_add_row(recipient_cluster_id)
                    self.correlograms_add_ACG(recipient_cluster_id)
                else:
                    self.summary_table_update_row(recipient_cluster_id)
                    self.correlograms_update_ACG_data(recipient_cluster_id)
                self._update_cluster_count_and_row_index()
                if add_or_edit == 'add':
                    # Add another graphical element for the new cluster
                    self._add_line_for_new_cluster(cluster_id=recipient_cluster_id, zorder=1)
                    visible_clusters.append(recipient_cluster_id)
                self.summary_table_update_zorder()
                # Reset plots
                self.update_plotted_data_of_cluster_id(recipient_cluster_id)
                for plot_name in self._plot_names:
                    self._update_plot(visible_clusters, plot_name, update_limits=False)
                self.currently_selected_clusters = np.array(visible_clusters, dtype=int)
                self._table_focus_on_cluster(visible_clusters)
                msg = 'Accepted selection'
                # A new selection will start
                if from_keyboard:
                    restart_selection = False
                else:
                    restart_selection = True
                # Update history
                self.history_manager(action='do')
            else:
                restart_selection = False
                msg = '(canceled polygon)'
            self.show_in_console_and_statusbar(msg)
        else:
            restart_selection = False
            msg = '(canceled polygon)'
            self.show_in_console_and_statusbar(msg)
        self.stop_polygon_selector()
        self.currently_selecting_spikes = False
        self.QTable.enable_interaction()
        self.set_all_buttons(to=True, except_button=add_or_edit)
        # Start a new selection
        if restart_selection:
            self.start_polygon_selector(add_or_edit, from_keyboard)

    def stop_polygon_selector(self):
        """Stop the selection with the polygon."""
        # Disconnect the polygon selector and reactivate the waveform selector
        self.callbacks.polygon_selectors.stability.disconnect()
        self.callbacks.polygon_selectors.feature.disconnect()
        # Remove selectors
        self.callbacks.polygon_selectors.stability = None
        self.callbacks.polygon_selectors.feature = None


    ############################################################################
    # User interaction with data
    ############################################################################
    def _merge_clusters(self, cluster_id_to_delete, cluster_id_to_expand):
        """Routine to perform to merge clusters."""
        # Make user input a list
        if not hasattr(cluster_id_to_delete, '__iter__'):
            cluster_id_to_delete = [cluster_id_to_delete]
        # For each cluster to merge ...
        for cluster_id in cluster_id_to_delete:
            # Get indices of spikes from cluster_id_to_delete
            cluster_spikes = self.get_spike_train(cluster_id)
            # Write the label of cluster_id_to_expand
            self.move_spikes_to_cluster(cluster_spikes, to_cluster_id=cluster_id_to_expand)
            # Cluster 0 cannot be removed
            if cluster_id == 0:
                self.summary_table_update_row(0)
                self.correlograms_update_ACG_data(0)
            else:
                # Remove cluster
                self._remove_cluster(cluster_id)
        self.summary_table_update_row(cluster_id_to_expand)
        self.correlograms_update_ACG_data(cluster_id_to_expand)
        # Focus on new cluster
        self.update_all_plots()
        self.show_only_cluster(cluster_id_to_expand)

    def _remove_cluster(self, cluster_id):
        """Routine to perform to delete a cluster."""
        # Row in table and index of graphic elements
        cluster_line = self.summary_table_get_row_index(cluster_id)
        # Delete all references to the graphical elements of this cluster
        for plt_name in self._plot_names_apart_waveform:
            self.ax[plt_name].lines.remove(self.SUMMARY_TABLE.loc[cluster_line, 'line_%s' % plt_name])
        [self.ax['waveform'][ch].lines.remove(self.SUMMARY_TABLE.loc[cluster_line, 'line_waveform'][ch]) for ch in range(self.n_channels_this_shank)]
        # Remove references from summary table
        self.SUMMARY_TABLE.drop(cluster_line, inplace=True)
        # Update table widget
        self._table_widget_remove_row(cluster_line)
        # Update the correlograms table
        self.correlograms_remove_cluster_id(cluster_id)
        # Update cluster count
        self._update_cluster_count_and_row_index()
        # Print a message to the log / console
        msg = '\tcluster %i has been removed' % cluster_id
        LOGGER.info(msg)


    def _table_widget_remove_row(self, cluster_line):
        """Update the QTable widget by removing a row."""
        # First of all, update internal tables
        self._update_cluster_count_and_row_index()
        # Remove row from QTableWidget
        self.QTable.remove_row(cluster_line)

    def _update_cluster_count_and_row_index(self):
        """Update number of clusters."""
        self.SUMMARY_TABLE.reset_index(drop=True, inplace=True)
        self.correlograms.reset_index(drop=True, inplace=True)
        self.n_clusters = self.SUMMARY_TABLE.shape[0]

    def summary_table_update_row(self, cluster_id, do_add=False):
        """Update / add info about a cluster in the SUMMARY_TABLE."""
        # Get the spike train for this cluster
        cluster_spikes = self.get_spike_train(cluster_id)
        n_spikes = cluster_spikes.shape[0]
        # Update spike count and firing rate
        table_index = self.summary_table_get_row_index(cluster_id)
        self.SUMMARY_TABLE.at[table_index, 'n_spikes'] = n_spikes
        self.SUMMARY_TABLE.at[table_index, 'FR'] = n_spikes / self.CS.INFO['recording_duration']
        self.SUMMARY_TABLE.reset_index(drop=True, inplace=True)
        table_row = self.QTable.get_row_containing(cluster_id)
        summary_table_row = self.summary_table_get_row_index(cluster_id)
        # Add / update row to QTable widget
        if do_add:
            self.QTable.add_row(self.SUMMARY_TABLE.loc[summary_table_row, :])
            # Connect callback function
            self.QTable.color_picker[table_row].clicked.connect(partial(self.callback_table_pick_a_color, cluster_id))
        else:
            # Simply update the row
            self.QTable.update_row(table_row, self.SUMMARY_TABLE.loc[summary_table_row])

    def summary_table_add_row(self, cluster_id):
        """Add a row to the summary table"""
        table_index = self.SUMMARY_TABLE.shape[0]
        self.SUMMARY_TABLE.at[table_index, 'id'] = cluster_id
        # self.SUMMARY_TABLE.at[table_index, 'color', tuple(np.random.rand(3, )))  # add random color
        self.SUMMARY_TABLE.at[table_index, 'color'] = next(self.CLUSTER_COLORS)
        self.SUMMARY_TABLE.at[table_index, 'cluster_type'] = 'm'
        self.SUMMARY_TABLE.at[table_index, 'z_order'] = 1
        self.summary_table_update_row(cluster_id, do_add=True)

    def summary_table_get_row_index(self, cluster_id):
        return self.SUMMARY_TABLE[self.SUMMARY_TABLE['id'] == cluster_id].index.values[0]

    def summary_table_update_zorder(self):
        # Sort the clusters by number of spikes and assign them a z-order so
        # that the more numerous clusters are plotted under smaller clusters and
        # don't obstruct them. The unsorted cluster is always plotted in the
        # background.
        self.SUMMARY_TABLE['z_order'] = np.hstack((np.array([1]), self.SUMMARY_TABLE.loc[1:, 'n_spikes'].argsort()[::-1] + 2))

    def make_new_cluster_id(self):
        """Find the lowest number that can be used as cluster id."""
        # Get the set of all existing ids on other channels
        already_existing = get_list_of_all_cluster_ids(self.CS.spikes_hdf5_filename, read_manually_refined_ids=True)
        # Add list of existing cluster ids on this channel
        already_existing = already_existing.union(set(self.SUMMARY_TABLE['id'].astype(int)))
        last_id = max(already_existing)
        # Get a missing number in the range from 1 to the maximum existing id
        possible_ids =  [i for i in range(1, last_id) if i not in already_existing]
        # If the list is empty, there is no number missing. Therefore, return the
        # maximum value + 1
        if len(possible_ids) == 0:
            return int(last_id + 1)
        # otherwise, return the first missing number
        else:
            return int(possible_ids[0])

    def get_spike_train(self, cluster_id):
        """Get the spike train of the selected cluster."""
        return np.where(self.DATA['cluster_id_manual'] == cluster_id)[0]

    def get_spike_waveforms(self, cluster_id, return_only_index=False):
        """Get waveforms of the selected cluster."""
        indices = self.DATA.loc[self.DATA['cluster_id_manual'] == cluster_id, 'index'].values
        if return_only_index:
            return indices
        else:
            return self.WAVEFORMS[indices, :, :]

    def move_spikes_to_cluster(self, cluster_spikes, to_cluster_id):
        self.DATA.loc[cluster_spikes, 'cluster_id_manual'] = to_cluster_id
        msg = '\t%i spikes were moved to cluster %i' % (cluster_spikes.shape[0], to_cluster_id)
        LOGGER.info(msg)

    def get_indices_of_points_in_selector(self):
        """Get the index of the points enclosed in the polygon."""
        points_selected_on_stability_plot = self.callbacks.polygon_selectors.stability.ind
        points_selected_on_feature_plot = self.callbacks.polygon_selectors.feature.ind
        ind = np.union1d(points_selected_on_stability_plot, points_selected_on_feature_plot).astype(int)
        spikes = self.callbacks.polygon_selectors.stability.points_id[ind]
        return ind.shape[0], spikes

    def get_cluster_colors_from_id(self, cluster_id_list):
        """Get the colors associated to the selected clusters."""
        if not hasattr(cluster_id_list, '__iter__'):
            cluster_id_list = [cluster_id_list]
        n_clusters = len(cluster_id_list)
        cluster_colors = np.empty((n_clusters, ), dtype=object)
        for idx, cluster_id in enumerate(cluster_id_list):
            row = self.summary_table_get_row_index(cluster_id)
            cluster_colors[idx] = self.SUMMARY_TABLE.loc[row, 'color']
        return list(cluster_colors)


    ############################################################################
    # User interaction with data - correlograms
    ############################################################################
    def compute_all_ACGs(self):
        """Compute the ACG of all clusters."""
        for idx in range(self.n_clusters):
            # Find the spikes in this cluster
            cluster_id = self.SUMMARY_TABLE.loc[idx, 'id']
            ACG = self._compute_ACG_of_cluster_id(cluster_id)
            # Append data to table
            last_row = self.correlograms.shape[0]
            self.correlograms.at[last_row, 'cluster_1'] = cluster_id
            self.correlograms.at[last_row, 'cluster_2'] = cluster_id
            self.correlograms.at[last_row, 'correlation'] = ACG
            self.correlograms.at[last_row, 'updated'] = True

    def _compute_ACG_of_cluster_id(self, cluster_id):
        """Get the spike train of cluster_id and compute its ACG."""
        cluster_spikes = self.get_spike_train(cluster_id)
        if cluster_spikes.shape[0] == 0:
            data = np.zeros_like(correlogram_make_lag_axis(self.CCG_max_lag, self.CCG_bin_size), dtype=float)
            return data
        else:
            spike_times = self.DATA['timestamp'][cluster_spikes].values / self.CS.INFO['sampling_frequency']
            spike_clusters = np.ones_like(spike_times, dtype=int)
            data = correlogram(spike_times, spike_clusters, self.CS.INFO['sampling_frequency'], self.CCG_bin_size, self.CCG_max_lag)
            return data[0, 0, :]  # [0, 0] contains the (1st and only) ACG

    def compute_CCGs_between_cluster_ids(self, cluster_id_list):
        """Computes the CCG between clusters."""
        # Make sure cluster_id_list is a list
        cluster_id_list = list(cluster_id_list)
        # Loop through the list of clusters
        all_spikes = np.empty((0, ), dtype=float)
        all_ids = np.empty_like(all_spikes)
        clusters_to_remove = list()
        for cluster_id in cluster_id_list:
            # Get the spike trains
            cluster_spikes = self.get_spike_train(cluster_id)
            if cluster_spikes.shape[0] == 0:
                clusters_to_remove.append(cluster_id)
                continue
            spike_times = self.DATA['timestamp'][cluster_spikes].values / self.OSSA.sampling_rate
            spike_id = np.ones_like(spike_times) * cluster_id
            # Concatenate the data
            all_spikes = np.concatenate((all_spikes, spike_times))
            all_ids = np.concatenate((all_ids, spike_id))
        # Remove empty clusters
        for cluster_id in clusters_to_remove:
            cluster_id_list.remove(cluster_id)
        # If only 1 cluster left, return negative outcome
        if len(cluster_id_list) < 2:
            return False

        # Compute the CCG
        data = correlogram(all_spikes, all_ids.astype(int, copy=False), self.OSSA.sampling_rate, self.CCG_bin_size, self.CCG_max_lag)
        # Because data is arranged in a matrix way, unravel the result and put it into the SUMMARY_TABLE
        ccg_idx_row, ccg_idx_col = np.triu_indices(len(cluster_id_list), k=1)
        # Go over each pair of cross-correlation histograms, ...
        for i, j in zip(ccg_idx_row.tolist(), ccg_idx_col.tolist()):
            cluster_id1 = cluster_id_list[i]
            cluster_id2 = cluster_id_list[j]
            # Get the row where these data are stored. If we haven't stored them
            # before, they get recomputed
            table_row = self.correlograms_get_row(cluster_id1, cluster_id2)
            self.correlograms.at[table_row, 'cluster_1'] = cluster_id1
            self.correlograms.at[table_row, 'cluster_2'] = cluster_id2
            self.correlograms.at[table_row, 'correlation'] = data[i, j, :]
            self.correlograms.at[table_row, 'updated'] = True
        # Update the table index
        self.correlograms.reset_index(drop=True, inplace=True)
        # Update the currently selected clusters
        self.currently_selected_clusters = np.array(cluster_id_list, dtype=int)
        # Return positive outcome
        return True

    def correlogram_make_x_data(self):
        """Make the lag axis for the correlograms."""
        # Get the first updated cluster
        up = np.where(self.correlograms['updated'])[0][0]
        data = self.correlograms.iloc[up]
        n_data_points = data['correlation'].shape[0]
        self.x_data['correlogram'] = np.linspace(-self.CCG_max_lag, self.CCG_max_lag, n_data_points)

    @staticmethod
    def correlogram_make_xy_outline(x, y):
        # Calculate the difference between x-values
        half_stepSize = np.diff(x) / 2.
        half_stepSize_before = np.concatenate(([half_stepSize[0]], half_stepSize))
        half_stepSize_after = np.concatenate((half_stepSize, [half_stepSize[-1]]))
        # Calculate the values for x and y values of the outline
        x = np.vstack((x-half_stepSize_before, x+half_stepSize_after)).transpose().ravel()
        y = np.tile(y, (2, 1)).transpose().ravel()
        return x, y

    def _reset_correlogram_xy_data(self):
        # Get only the lag axis
        self.x_data['correlogram'] = correlogram_make_lag_axis(self.CCG_max_lag, self.CCG_bin_size)
        self.y_data['correlogram'] = np.zeros((self.x_data['correlogram'].shape[0], ), dtype=float)

    def correlogram_make_xlims(self):
        """Make ticks and labels of correlogram from its x-limits"""
        x_lims = compute_array_range(self.x_data['correlogram'], padding=False)
        # Set and save plot limits
        self.set_plot_lims(x=np.array(x_lims), y=np.empty(0), on_plot='correlogram')

    def correlograms_get_data(self, cluster_id1, cluster_id2):
        """Get the correlogram between 2 clusters."""
        row = self.correlograms_get_row(cluster_id1, cluster_id2)
        data = self.correlograms.loc[row, 'correlation']
        return data

    def correlograms_get_row(self, cluster_id1, cluster_id2=None):
        """Get the row of the correlogram table from a pair of cluster ids"""
        if cluster_id2 is None:  # Return all the pairs in which cluster_id1 is in either the first or the second column
            ind = np.where(np.logical_or(self.correlograms['cluster_1'] == cluster_id1, self.correlograms['cluster_2'] == cluster_id1))[0]
        else:  # return a specific pair
            ind = np.where(np.logical_and(self.correlograms['cluster_1'] == cluster_id1, self.correlograms['cluster_2'] == cluster_id2))[0]

        if len(ind) == 0:  # If there is no corresponding data, return the row where new data should be placed
            return int(self.correlograms.shape[0])
        elif len(ind) == 1:  # If only one hit, return it as int
            return int(ind[0])
        else:  # Return an array of ints
            return ind.astype(int)

    def correlograms_remove_cluster_id(self, cluster_id):
        """Remove the correlogram of a cluster."""
        row = self.correlograms_get_row(cluster_id)
        self.correlograms.drop(self.correlograms.index[row], inplace=True)
        self.correlograms.reset_index(drop=True, inplace=True)

    def correlograms_update_ACG_data(self, cluster_id):
        """Update the ACG data for the selected cluster."""
        ACG = self._compute_ACG_of_cluster_id(cluster_id)
        row_ACG = self.correlograms_get_row(cluster_id, cluster_id)  # the ACG of this cluster
        self.correlograms.at[row_ACG, 'correlation'] = ACG
        self.correlograms.at[row_ACG, 'updated'] = True
        # Mark CCGs with this cluster as not updated, but we don't recompute them now
        rows_CCG = self.correlograms_get_row(cluster_id)
        if not hasattr(rows_CCG, '__iter__'):  # There was only onw row for this cluster, i.e. the ACG
            return
        rows_CCG = rows_CCG[rows_CCG != row_ACG]
        self.correlograms.loc[rows_CCG, 'updated'] = False

    def correlograms_update_ACG_plot(self, cluster_id):
        cluster_line = self.summary_table_get_row_index(cluster_id)
        y = self.correlograms_get_data(cluster_id, cluster_id)
        # Make outline data
        x, y = self.correlogram_make_xy_outline(self.x_data['correlogram'], y)
        # Set new data
        self.SUMMARY_TABLE.loc[cluster_line, 'line_correlogram'].set_data(x, y)

    def correlograms_add_ACG(self, cluster_id):
        """Add an ACG when adding a cluster."""
        ACG = self._compute_ACG_of_cluster_id(cluster_id)
        # Append data to table
        last_row = self.correlograms.shape[0]
        self.correlograms.at[last_row, 'cluster_1'] = cluster_id
        self.correlograms.at[last_row, 'cluster_2'] = cluster_id
        self.correlograms.at[last_row, 'correlation'] = ACG
        self.correlograms.at[last_row, 'updated'] = True
        self.correlograms.reset_index(drop=True, inplace=True)

    def _plot_CCG(self, cluster_id_list):
        """Open a new window and plot the CCG between clusters."""
        # Get data dimensions
        n_clusters = cluster_id_list.shape[0]
        if n_clusters < 2:
            return

        # Open figure and make a grid of subplots
        style_dict = set_figure_style(self.OSSA.default['axes_color'])
        title_color = style_dict['axes.labelcolor']
        fig, ax = plt.subplots(nrows=n_clusters, ncols=n_clusters, sharex='all')
        ax = np.fliplr(ax)  # So ACG are on anti-diagonal
        # Start hiding all axes
        [ax[i, j].set_visible(False) for i in np.arange(len(cluster_id_list)) for j in np.arange(len(cluster_id_list))]
        # Get the original colors for these clusters
        cluster_colors = self.get_cluster_colors_from_id(cluster_id_list)
        # Get the x-values of the histograms
        self._reset_correlogram_xy_data()
        x_data = self.x_data['correlogram']

        # Plot CCGs only once, in the upper-triangle of the array. Find indices of
        # these plots
        ccg_idx_row, ccg_idx_col = np.triu_indices(n_clusters)
        pairs = list(zip(ccg_idx_row.tolist(), ccg_idx_col.tolist()))

        # Go over each pair of cross-correlation histograms, ...
        for p in range(len(pairs)):
            i = pairs[p][0]
            j = pairs[p][1]
            # Get CCG data
            cluster_id1 = cluster_id_list[i]
            cluster_id2 = cluster_id_list[j]
            table_row = self.correlograms_get_row(cluster_id1, cluster_id2)
            data = self.correlograms.loc[table_row, 'correlation']

            # Get cluster color for ACG; use black for CCGs
            color = cluster_colors[i] if i == j else np.zeros(3)

            # Draw bars
            ax[i, j].set_visible(True)
            ax[i, j].bar(left=x_data, height=data, width=self.CCG_bin_size, bottom=0, color=color, edgecolor='None', linewidth=0.0)

            # Add the cluster id number above the first row and on the left of the
            # first column
            if i == 0:
                ax[i, j].set_title('cluster %i' % cluster_id_list[j], fontdict={'fontsize': 14}, color=title_color)
            if j == n_clusters - 1:
                ax[i, j].set_ylabel('cluster %i' % cluster_id_list[i], fontdict={'fontsize': 14})
            # Add the cluster label beside each ACG
            if i == j and i != 0 and j != 0 and i != n_clusters - 1 and j != n_clusters - 1:
                ax[i, j].set_ylabel('cluster %i' % cluster_id_list[i], fontdict={'fontsize': 14})
                ax[i, j].yaxis.set_label_position('right')
                ax[i, j].yaxis.label.set_rotation(270)
                ax[i, j].yaxis.label.set_verticalalignment('bottom')

            # Fix y-lims, remove all y-tick labels but leave the grid on
            y_range = [0, compute_array_range(data, padding=True)[1] * 1.5]
            ax[i, j].set_ylim(y_range)
            ax[i, j].set_yticks(list())
            ax[i, j].set_yticklabels(list())

        # Restore the visibility of the x-axes in ACGs (plt.subplots() hides them when axes limits are shared)
        for i in range(n_clusters):
            for label in ax[i, i].get_xticklabels():
                label.set_visible(True)
            ax[i, i].xaxis.offsetText.set_visible(True)
            ax[i, i].tick_params(axis='x', colors=title_color)

        # Fix axes appearance
        ax[-1, -1].set_xlabel('lag (ms)')
        ax[0, 0].set_xlim((-self.CCG_max_lag, self.CCG_max_lag))

        # Remove unused axes and empty space
        ax = ax.ravel().tolist()
        empty_axes_idx = np.where([not ax[i].get_visible() for i in range(len(ax))])[0]
        [fig.delaxes(ax[i]) for i in empty_axes_idx]
        # Apply  tight layout
        fig.tight_layout(pad=1.08, h_pad=.5, w_pad=.5, rect=[0, 0, 1, 1])
        # Enable the panning tool by default
        fig.canvas.toolbar.pan()
        # Show the figure
        fig.show()


    ############################################################################
    # Inter-spike interval distribution
    ############################################################################
    def _compute_ISI(self, cluster_id_list):
        """Computes the ISIs of cluster_ids in cluster_id_list."""
        ISI = list()
        for cluster_id in cluster_id_list:
            # Get spike timestamps list from this cluster_id
            cluster_spikes = self.get_spike_train(cluster_id)
            spikes = self.DATA['timestamp'][cluster_spikes] / self.CS.INFO['sampling_frequency']
            # Convert to ms
            spikes *= 1000.
            # Compute inter-spike interval
            ISI.append(np.diff(spikes))
        # Get maximum ISI
        max_isi = np.array([i.max() for i in ISI]).max()
        # Make bin edge array
        bin_array = np.arange(0, max_isi, step=self.OSSA.analysis['ISI_bin_size'])
        # Compute distributions
        ISId = np.zeros((cluster_id_list.shape[0], bin_array.shape[0]-1), dtype=float)
        for ind in range(len(cluster_id_list)):
            # Bin data
            d = np.histogram(ISI[ind], bin_array)[0].astype(float)
            # Normalize to area = 100%
            d = d / (ISI[ind].shape[0] + 1) * 100.
            # Store data
            ISId[ind, :] = d
        # Return output
        return ISId, bin_array

    def compute_and_plot_ISI(self, cluster_id_list):
        """Open a new window and plot the ISI distribution of the selected clusters."""
        # Check that at least one cluster has at least 2 spikes in it
        good_clusters = list()
        for cluster_id in cluster_id_list:
            row = self.summary_table_get_row_index(cluster_id)
            n_spikes = self.SUMMARY_TABLE.loc[row, 'n_spikes']
            if n_spikes >= 2:
                good_clusters.append(cluster_id)
        # Re-transform to numpy-array
        cluster_id_list = np.array(good_clusters, dtype=int)

        # Get number of clusters
        n_clusters = cluster_id_list.shape[0]
        if n_clusters < 1:
            return
        # Compute ISIs
        ISId, bin_array = self._compute_ISI(cluster_id_list)
        # Change alpha of the plot according to whether there is 1 or more clusters
        if n_clusters > 1:
            plot_alpha = 0.7
        else:
            plot_alpha = 1.0
        # Make plot outline
        X_toplot = None
        Y_toplot = None
        for ind in range(len(cluster_id_list)):
            x, y = self.correlogram_make_xy_outline(bin_array[:-1], ISId[ind, :])
            if ind == 0:
                X_toplot = x + (self.OSSA.analysis['ISI_bin_size'] / 2.)
                Y_toplot = np.atleast_2d(y)
            else:
                Y_toplot = np.vstack((Y_toplot, y))

        # Open figure
        set_figure_style(self.OSSA.default['axes_color'])
        fig, ax = plt.subplots(1, 1)
        # Plot clusters one by one to preserve colors
        colors = self.get_cluster_colors_from_id(cluster_id_list)
        for ind, cluster_id in enumerate(cluster_id_list):
            ax.fill_between(X_toplot, Y_toplot[ind, :], y2=0, where=Y_toplot[ind, :]>0, facecolor=colors[ind], edgecolor='None', alpha=plot_alpha)
        # Plot a vertical line for the threshold set by the user in ROSS_configs
        ax.axvline(self.OSSA.analysis['ISI_threshold'], color='r', linewidth=2)
        # Make log-scale
        ax.set_xscale('log', nonposx='clip')
        # Set x-axis limits, and add labels
        ax.set_xlim((ax.get_xlim()[0], 1000))
        ax.set_xlabel('Inter-spike interval (ms)')
        ax.set_ylabel('% of all spikes')

        # For each cluster, find the percent of spikes below the threshold (RPV
        # stands for Refractory-Period Violations)
        threshold_x = np.argmin(np.abs(bin_array[:-1] - self.OSSA.analysis['ISI_threshold'])) - 1
        RPV = np.sum(ISId[:, :threshold_x], axis=1)
        # Write threshold in the figure title
        fig.suptitle('ISI threshold: %.1f ms' % self.OSSA.analysis['ISI_threshold'])
        # Print info in graph and console
        text = ""
        for ind, cluster_id in enumerate(cluster_id_list):
            text += 'cluster %i - %.1f' % (cluster_id, RPV[ind]) + r'%' + '\n'
        ax.annotate(text, xy=(.99, .99), xytext=(.99, .99), xycoords='axes fraction', ha='right', va='top', clip_on=True, fontsize=12)
        self.show_in_console_and_statusbar(text.replace('\n', ' | ')[:-2])
        # Apply tight layout
        fig.tight_layout(pad=1.08, h_pad=.5, w_pad=.5, rect=[0, 0, 1, 1])
        # Enable the panning tool by default
        fig.canvas.toolbar.pan()
        # Show figure
        fig.show()


    ############################################################################
    # Callback functions
    ############################################################################
    def callback_on_resize(self):
        for plt_name in self._plot_names_apart_waveform:
            try:
                self.fig[plt_name].tight_layout(pad=0)
                self.ax[plt_name].figure.canvas.draw_idle()
            except ValueError:  # This happens when the axes is squeezed too much and a new axes-range cannot be computed
                pass  # Silently ignore it
        try:
            [self.fig['waveform'][ch].tight_layout(pad=0) for ch in range(self.n_channels_this_shank)]
            [self.ax['waveform'][ch].figure.canvas.draw_idle() for ch in range(self.n_channels_this_shank)]
        except ValueError:  # This happens when the axes is squeezed too much and a new axes-range cannot be computed
            pass  # Silently ignore it

    def callback_show_clicked_waveform(self, event, from_plot):
        """Callback to show and update the waveform of the clicked spike."""
        # Check whether waveform picking is allowed
        if not self.is_waveform_picking_allowed(from_plot):
            return
        # Get number of points around cursor and keep a random one in the area
        # surrounding the clicked point
        N_points = len(event.ind)
        the_clicked_point = np.random.choice(N_points)
        data_point_index = event.ind[the_clicked_point]
        # Find out which axis the user clicked on
        clicked_cluster_id = self._get_clicked_plot_info(event.artist)
        # Get index of point in data
        cluster_spikes = self.get_spike_train(clicked_cluster_id)
        real_data_point_index = cluster_spikes[data_point_index]
        # Update waveform plot
        y_data = self.WAVEFORMS[self.DATA.loc[real_data_point_index, 'index'], :, :].transpose().tolist()
        [self.last_selected_line.waveform[ch].set_ydata(y_data[ch]) for ch in range(self.n_channels_this_shank)]
        # Re-compute limits
        for cluster_id in self.currently_selected_clusters:
            cluster_line = self.summary_table_get_row_index(cluster_id)
            y_data = np.concatenate((y_data, [self.SUMMARY_TABLE.loc[cluster_line, 'line_waveform'][ch].get_ydata() for ch in range(self.n_channels_this_shank)]))
        ylims_waveform = compute_array_range(y_data, padding=True)
        self.set_plot_lims(x=np.empty(0), y=ylims_waveform, on_plot='waveform')
        # Update last selected point position
        x_stability, x_stability_relative, y_stability, x_feature, y_feature = self.DATA.loc[real_data_point_index, ['timestamp', 'time', self.OSSA.plotted_features['SpikeSorter']['y']['stability'], self.OSSA.plotted_features['SpikeSorter']['x']['feature'], self.OSSA.plotted_features['SpikeSorter']['y']['feature']]]
        color = self.get_cluster_colors_from_id([clicked_cluster_id])[0]
        self.update_last_selected_point(x_stability, y_stability, x_feature, y_feature, color)
        # Update info on selected point in stability plot
        tooltip_base = translate_feature_list_to_tooltip('timestamp')
        self.text.stability.set_text(tooltip_base % x_stability_relative)
        # Update canvases
        for plt_name in ['stability', 'feature']:
            self.ax[plt_name].figure.canvas.draw_idle()
        [self.ax['waveform'][ch].figure.canvas.draw_idle() for ch in range(self.n_channels_this_shank)]

    def callback_reset_zoom(self):
        """Reset limits on all plots"""
        for plt_name in self._plot_names:
            self._update_plot(self.currently_selected_clusters, plt_name, update_data=False, update_limits=True)
            if plt_name == 'waveform':
                [self.ax['waveform'][ch].figure.canvas.draw() for ch in range(self.n_channels_this_shank)]
            else:
                self.ax[plt_name].figure.canvas.draw()

    def callback_pan_zoom_figure(self):
        """Toggle the state of the panning tool."""
        self.toggle_button_pan_zoom(not self.currently_panning)

    def callback_quit(self, save):
        """Routine to perform when hitting the 'save' button."""
        if save:  # Save current state of the user selection
            msg = 'Saving data to disk'
            LOGGER.info(msg)
            # Store the cluster_types
            cluster_types = list(np.empty((self.n_clusters,), dtype=object))
            for idx in range(self.n_clusters):
                cluster_id = int(self.SUMMARY_TABLE.loc[idx, 'id'])
                table_row = self.QTable.get_row_containing(cluster_id)
                qTableItem = self.QTable.item(table_row, 4)
                if qTableItem is not None:  # for the 0 cluster
                    cluster_type = str(qTableItem.text())
                else:
                    qTableItem = self.QTable.cellWidget(table_row, 4)  # all other clusters
                    cluster_type = str(qTableItem.currentText())
                cluster_types[idx] = cluster_type[0].lower()
            # Storing current colors
            colors = self.SUMMARY_TABLE['color'].values
            # Reset data in memory (to update number of spikes)
            self.reset_summary_table(from_disk=False)
            # Restore cluster types and colors
            self.SUMMARY_TABLE['cluster_type'] = cluster_types
            self.SUMMARY_TABLE['color'] = colors

            # Find 'noise' clusters, apart from 0
            noise_clusters = list(self.SUMMARY_TABLE.loc[self.SUMMARY_TABLE['cluster_type']=='n', 'id'].astype(int))
            noise_clusters.remove(0)
            # Read current cluster IDs
            cluster_id_array = read_hdf5_data(self.CS.spikes_hdf5_filename, table_name=self.CS.shank_selection_names, column_list='cluster_id_manual')
            # Set all to 0
            cluster_id_array = np.zeros_like(cluster_id_array)
            # Update cluster ids of shown spikes
            cluster_id_array[self.DATA['index'].values] = self.DATA['cluster_id_manual'].values

            if len(noise_clusters) > 0:
                # Loop through each one of them, and move these spikes to cluster 0
                for n_id in noise_clusters:
                    # Fix cluster_id
                    cluster_id_array[cluster_id_array == n_id] = 0

                # Remove rows corresponding to noisy cluster
                self.SUMMARY_TABLE.drop(np.where(np.in1d(self.SUMMARY_TABLE['id'], noise_clusters))[0], inplace=True)
                self.SUMMARY_TABLE.reset_index(drop=True, inplace=True)

            # Read current 'good spike' flags
            good_spike_array = read_hdf5_data(self.OSSA.spikes_hdf5_filename, table_name=self.CS.shank_selection_names, column_list='good_spike')
            # Find number of spikes for cluster 0
            n_spikes = np.sum(np.logical_and(cluster_id_array == 0, good_spike_array))
            # Update n_spikes for cluster 0
            row_0 = np.where(self.SUMMARY_TABLE['id'] == 0)[0][0]
            self.SUMMARY_TABLE.at[row_0, 'n_spikes'] = n_spikes
            self.SUMMARY_TABLE.at[row_0, 'FR'] = n_spikes / self.OSSA.recording_duration

            # Update SUMMARY TABLE on disk
            write_sorting_summary_table(self.OSSA.spikes_hdf5_filename, self.CS.shank_selection_names, self.SUMMARY_TABLE)

            # Check whether there this channel contains good spikes
            cluster_types = list(self.SUMMARY_TABLE['cluster_type'])
            contains_spikes = 's' in cluster_types or 'm' in cluster_types
            # Update OSSA_results_table on disk
            self.OSSA.update_OSSA_results(self.CS.shank_selection_names, 'SpikeSorter', contains_spikes)

            # Write cluster IDs to disk
            update_hdf5_data(self.OSSA.spikes_hdf5_filename, table_name=self.CS.shank_selection_names, column='cluster_id_manual', values=cluster_id_array)

            msg = '(ok)'
            LOGGER.info(msg)
            # Assign exit code
            self.OSSA.exit_code = 2
        else:
            msg = 'Discarding changes'
            LOGGER.info(msg)
            # Assign exit code
            self.OSSA.exit_code = 0

        # Close this GUI
        msg = 'Quitting GUI and going back to ChannelSelector'
        LOGGER.info(msg)
        self.on_close()
        msg = '(ok)'
        LOGGER.info(msg)

    def callback_reload(self):
        self.history_manager(action='reset')

    def callback_change_plotted_features(self, plot_name, feature):
        if plot_name == 'stability_y':
            self.OSSA.plotted_features['SpikeSorter']['y']['stability'] = feature
            self.ax.stability.set_ylabel(translate_feature_to_axis_label(feature))
            self._update_plot(self.currently_selected_clusters, 'stability', update_limits=True)

        elif plot_name == 'feature_x':
            self.OSSA.plotted_features['SpikeSorter']['x']['feature'] = feature
            self.ax.feature.set_xlabel(translate_feature_to_axis_label(feature))
            self._update_plot(self.currently_selected_clusters, 'feature', update_limits=True)

        elif plot_name == 'feature_y':
            self.OSSA.plotted_features['SpikeSorter']['y']['feature'] = feature
            self.ax.feature.set_ylabel(translate_feature_to_axis_label(feature))
            self._update_plot(self.currently_selected_clusters, 'feature', update_limits=True)

    def callback_table_new_selection(self, update_limits=True):
        # This is True only when user clicks around. During background operations
        # that might change the selection, the flag is turned off so that figure
        # updates are not fired off accidentally.
        if self.QTable.allowed_interaction:
            items = np.array(self.QTable.selectedItems(), dtype=object)
            if items.shape[0] == 0:
               items = np.array([self.QTable.item(jj, 1) for jj in [self.summary_table_get_row_index(ii) for ii in self.currently_selected_clusters]], dtype=object)
            else:
                self.currently_selected_clusters = self.QTable.return_from_column(0, items, data_type=int)
            rows = self.QTable.get_row_number(items)
            self.update_button_state_from_table_selection(rows)
            self.update_figures(update_limits=update_limits)

    def callback_table_pick_a_color(self, cluster_id):
        """Routine to select a new color."""
        self.QTable.disable_interaction()
        # Show color picker
        color_picker = QtWidgets.QColorDialog()
        # Extract RGB values in range [0 1]
        color = color_picker.getColor()
        if color.isValid():
            color = color.getRgbF()[:3]
            # Get the row in the table
            table_row = self.QTable.get_row_containing(cluster_id)
            # Change color of button
            color_Qt_button(self.QTable.color_picker[table_row], color)
            # Change color of corresponding lines
            table_row = self.summary_table_get_row_index(cluster_id)
            self.SUMMARY_TABLE.loc[table_row, 'line_stability'].set_markerfacecolor(color)
            self.SUMMARY_TABLE.loc[table_row, 'line_feature'].set_markerfacecolor(color)
            self.SUMMARY_TABLE.loc[table_row, 'line_waveform'].set_color(color)
            self.SUMMARY_TABLE.loc[table_row, 'line_correlogram'].set_color(color)
            # Update color in main table
            self.SUMMARY_TABLE.at[table_row, 'color'] = color
            # Redraw canvases
            for plt_name in self._plot_names_apart_waveform:
                self.ax[plt_name].figure.canvas.draw_idle()
            [self.ax['waveform'][ch].figure.canvas.draw_idle() for ch in range(self.n_channels_this_shank)]
            # Update history
            self.history_manager(action='do')
        self.QTable.enable_interaction()

    def callback_table_change_cluster_type(self, cluster_id):
        """Routine to change unit type in memory."""
        # Get new value
        table_row = self.QTable.get_row_containing(cluster_id)
        value = self.QTable.cluster_type_picker[table_row].currentIndex()
        combo_box_options = ['noise', 'SU', 'MU']
        value_str = combo_box_options[value][0].lower()
        # Update table in memory
        table_row = self.summary_table_get_row_index(cluster_id)
        self.SUMMARY_TABLE.at[table_row, 'cluster_type'] = value_str
        # Update history
        self.history_manager(action='do')

    def callback_add_cluster(self, from_keyboard=False):
        # Do something only if button is currently enabled
        if self.menu.add_cluster.isEnabled():
            if self.currently_selecting_spikes:
                if not from_keyboard:
                    self.validate_polygon_selection('add')
                else:  # Ignore invoking keyboard shortcut again. Use appropriate command to terminate the selection.
                    pass
            else:
                msg = 'Selecting spikes for new cluster'
                self.show_in_console_and_statusbar(msg)
                self.start_polygon_selector('add', from_keyboard)

    def callback_edit_cluster(self, from_keyboard=False):
        # Do something only if button is currently enabled
        if self.menu.edit_cluster.isEnabled():
            if self.currently_selecting_spikes:
                if not from_keyboard:
                    self.validate_polygon_selection('edit')
                else:  # Ignore invoking keyboard shortcut again. Use appropriate command to terminate the selection.
                    pass
            else:
                msg = 'Selecting spikes to remove from all clusters'
                self.show_in_console_and_statusbar(msg)
                self.start_polygon_selector('edit', from_keyboard)

    def callback_delete_cluster(self):
        """When clicking on 'delete cluster' button."""
        # Remove cluster 0 if it is currently selected
        self.currently_selected_clusters = self.currently_selected_clusters[self.currently_selected_clusters != 0]
        if self.currently_selected_clusters.shape[0] == 0:  # no selected clusters
            return
        msg = 'Removing cluster(s): %s' % self.currently_selected_clusters
        self.show_in_console_and_statusbar(msg)
        self.QTable.disable_interaction()
        self._merge_clusters(cluster_id_to_delete=self.currently_selected_clusters, cluster_id_to_expand=0)
        # Update history
        self.history_manager(action='do')
        self._table_focus_on_cluster([0])
        self.QTable.enable_interaction()
        self.show_in_console_and_statusbar('(ok)')

    def callback_merge_clusters(self):
        """When clicking on 'merge cluster' button."""
        self.show_in_console_and_statusbar('Merging cluster(s): %s' % self.currently_selected_clusters)
        self.QTable.disable_interaction()
        # By default, we merge the smaller clusters into the largest one. However,
        # if the cluster '0' is selected, it should never be used as the recipient
        # cluster.
        # Get number of spikes in each selected cluster
        n_spikes = np.zeros((len(self.currently_selected_clusters), ), dtype=int)
        for idx, cluster_id in enumerate(self.currently_selected_clusters):
            n_spikes[idx] = self.SUMMARY_TABLE.loc[self.summary_table_get_row_index(cluster_id), 'n_spikes']
        # Isolate the biggest cluster
        n_spikes_order = np.argsort(n_spikes)
        sorted_clusters = self.currently_selected_clusters[n_spikes_order]
        large_cluster = sorted_clusters[-1]
        if large_cluster == 0:
            large_cluster = sorted_clusters[-2]
        # Merge the small clusters into the larger one
        clusters_to_delete = self.currently_selected_clusters[self.currently_selected_clusters != large_cluster]
        self._merge_clusters(cluster_id_to_delete=clusters_to_delete, cluster_id_to_expand=large_cluster)
        # Update history
        self.history_manager(action='do')
        self._table_focus_on_cluster(large_cluster)
        self.QTable.enable_interaction()
        self.show_in_console_and_statusbar('(ok)')

    def callback_recompute_ACG(self):
        """When changing bin_size or lag in text edit fields."""
        # Check that values are not empty
        try:
            new_max_lag = int(self.textEdits.edit_ACG_lag.text())
            new_bin_size = int(self.textEdits.edit_ACG_bin.text())
        except ValueError:
            self.textEdits.edit_ACG_lag.setText(str('%i' % self.CCG_max_lag))
            self.textEdits.edit_ACG_bin.setText(str('%i' % self.CCG_bin_size))
            return
        # Bin size cannot be bigger than window width. If so, restore previous
        # values and ignore user input
        if new_bin_size > new_max_lag:
            self.textEdits.edit_ACG_lag.setText(str('%i' % self.CCG_max_lag))
            self.textEdits.edit_ACG_bin.setText(str('%i' % self.CCG_bin_size))
            return
        # Copy new values in memory and make sure the window width is a multiple
        # of the bin size and an odd number
        n_bins = float(np.ceil(new_max_lag / float(new_bin_size)))
        n_bins = int(np.floor(n_bins / 2.) * 2. + 1)
        # Make sure number of bins is at least 5
        if n_bins < 5:
            n_bins = 5
        new_max_lag = n_bins * new_bin_size
        self.CCG_max_lag = new_max_lag
        self.CCG_bin_size = new_bin_size
        # Update values in the GUI
        self.textEdits.edit_ACG_lag.setText(str('%i' % self.CCG_max_lag))
        self.textEdits.edit_ACG_bin.setText(str('%i' % self.CCG_bin_size))
        # Log outcome
        msg = 'Recomputing ACGs with bin_size=%ims and max_lag=%ims' % (self.CCG_bin_size, self.CCG_max_lag)
        self.show_in_console_and_statusbar(msg)
        # Recompute all ACGs
        for idx in range(self.n_clusters):
            cluster_id = self.SUMMARY_TABLE.loc[idx, 'id']
            self.correlograms_update_ACG_data(cluster_id)
        # Update plots
        self._reset_correlogram_xy_data()
        self.correlogram_make_xlims()
        for idx in range(self.n_clusters):
            cluster_id = self.SUMMARY_TABLE.loc[idx, 'id']
            self.correlograms_update_ACG_plot(cluster_id)
        # It can happen that the lines that are visible are not currently
        # selected in the table. Then, check which clusters are currently visible
        # assuming that the user is referring to what he/she sees plotted and not
        # what is currently selected in the table
        self.currently_selected_clusters = np.empty((0, ))
        for idx in range(self.n_clusters):
            if self.SUMMARY_TABLE.loc[idx, 'line_stability'].get_visible():
                cluster_id = self.SUMMARY_TABLE.loc[idx, 'id']
                self.currently_selected_clusters = np.concatenate((self.currently_selected_clusters, [cluster_id]))
        self.currently_selected_clusters.astype(int).sort()
        self._update_plot(self.currently_selected_clusters, 'correlogram', update_data=False, update_limits=True)
        self.show_in_console_and_statusbar('(ok)')

    def callback_compute_ISI_distributions(self):
        """When clicking on 'ISI' button."""
        self.currently_selected_clusters.sort()
        msg = 'Computing ISI of clusters %s' % self.currently_selected_clusters
        self.show_in_console_and_statusbar(msg)
        self.compute_and_plot_ISI(self.currently_selected_clusters)
        self.show_in_console_and_statusbar('(ok)')

    def callback_force_clean_ISI(self):
        """Force remove all spikes that fall in refractory period. Spurious spikes
        are removed by comparing putative errors with the mean waveform. Spikes
        are moved to a new cluster for further analysis.
        """
        # This script works with only 1 cluster selected, but <currently_selected_clusters>
        # is an array, so extract its (only 1) value.
        cluster_id = self.currently_selected_clusters[0]
        # Get the spikes for this cluster
        cluster_spikes = self.get_spike_train(cluster_id)
        # Get timestamps in ms
        timestamps = self.DATA.loc[cluster_spikes, 'timestamp'].values / self.OSSA.sampling_rate * 1000.
        # Get ISI and spikes below the threshold
        ISI = np.diff(timestamps)
        bad_isi = np.where(ISI < self.OSSA.analysis['ISI_threshold'])[0]
        if bad_isi.shape[0] > 0:
            # Take the waveforms of this cluster
            waveforms = self.get_spike_waveforms(cluster_id)

            # Split the array of bad spikes in a list of contiguous values
            bad_isi_list = np.split(bad_isi, np.where(np.diff(bad_isi) != 1)[0] + 1)
            # Append index of last element to each group of the list
            bad_isi_list = [list(i) + [i[-1]+1] for i in bad_isi_list]
            # Re-concatenate list of indices
            bad_isi = np.array(list(chain.from_iterable(bad_isi_list)), dtype=int)

            # If all spikes somehow are 'bad', log and return
            if np.array_equal(bad_isi, cluster_spikes):
                msg = 'WARNING: All spikes in cluster %i violate the refractory period of %.1f ms' % (cluster_id, self.OSSA.analysis['ISI_threshold'])
                self.show_in_console_and_statusbar(msg)
                return

            # Get the mean waveform of the spikes that are still in the original cluster
            good_idx = np.logical_not(np.in1d(np.arange(waveforms.shape[0]), bad_isi))
            mean_waveform = np.mean(waveforms[good_idx, :], axis=0)

            # Go though each group of spikes and check which ones are more similar
            # to the mean waveform of the main cluster.
            index_bad_isi = list()
            index_good_isi = list()
            for ind in bad_isi_list:
                # Calculate the total sum of squares of each waveform
                relative_difference = np.sum((mean_waveform - waveforms[ind, :]) ** 2, axis=1)
                # Get the spike with the smallest difference from the mean
                good_spike = relative_difference.argmin()
                index_good_isi.append(ind[good_spike])
                # Get all the other spikes
                index_bad_isi += list(np.delete(ind, good_spike))
            # Sort list of duplicates and make sure they contain integers
            index_good_isi.sort()
            index_good_isi = np.array(index_good_isi, dtype=int)
            index_bad_isi.sort()
            index_bad_isi = np.array(index_bad_isi, dtype=int)

            # Make new clusters for each bad ISI group
            bad_cluster_id = good_cluster_id = 0
            for i in range(2):
                # Make a new cluster id value
                new_cluster_id = self.make_new_cluster_id()
                if i == 0:
                    idx = index_good_isi
                    good_cluster_id = new_cluster_id
                else:
                    idx = index_bad_isi
                    bad_cluster_id = new_cluster_id
                # Re-label these spikes
                self.move_spikes_to_cluster(cluster_spikes[idx], to_cluster_id=new_cluster_id)
                # Add a row in the SUMMARY TABLE
                self.summary_table_add_row(new_cluster_id)
                # Compute ACG
                self.correlograms_add_ACG(new_cluster_id)
                # Add graphical element
                self._add_line_for_new_cluster(cluster_id=new_cluster_id, zorder=1)

            # Reset plots
            self._update_cluster_count_and_row_index()
            self.summary_table_update_zorder()
            visible_clusters = [cluster_id, good_cluster_id, bad_cluster_id]
            for plot_name in self._plot_names:
                self._update_plot(visible_clusters, plot_name, update_limits=True)
            self.currently_selected_clusters = np.array(visible_clusters, dtype=int)

            # Show message in log, console and statusbar
            msg = 'Made clusters %i and %i from %i good and %i bad spikes violating the refractory period of %.1f ms in cluster %i' % (good_cluster_id, bad_cluster_id, index_good_isi.shape[0], index_bad_isi.shape[0], self.OSSA.analysis['ISI_threshold'], cluster_id)
        else:
            msg = 'Cluster %i has no spikes violating the refractory period of %.1f ms' % (cluster_id, self.OSSA.analysis['ISI_threshold'])
        self.show_in_console_and_statusbar(msg)

    def callback_compute_CCGs(self):
        """When clicking on 'CCG' button."""
        self.currently_selected_clusters.sort()
        msg = 'Computing CCGs of clusters %s' % self.currently_selected_clusters.astype(int)
        self.show_in_console_and_statusbar(msg)
        outcome = self.compute_CCGs_between_cluster_ids(self.currently_selected_clusters)
        if outcome:
            self._plot_CCG(self.currently_selected_clusters)
            self.show_in_console_and_statusbar('(ok)')

    def callback_compute_PCs(self):
        """Recompute the PCs of the selected clusters."""
        self.show_in_console_and_statusbar('Recomputing PCs of clusters %s' % self.currently_selected_clusters)
        # Gather all waveforms of interest
        indices = np.empty((0, ), dtype=int)
        all_spike_indices = np.empty((0, ), dtype=int)
        for cluster_id in self.currently_selected_clusters:
            cluster_spikes = self.get_spike_train(cluster_id)
            if cluster_spikes.shape[0] == 0:
                continue
            wave_idx = self.DATA.loc[cluster_spikes, 'index']
            all_spike_indices = np.concatenate((all_spike_indices, cluster_spikes))
            indices = np.concatenate((indices, wave_idx))
        # Get waveforms corresponding to spikes of interest
        w = self.WAVEFORMS[indices, :, :]
        w = np.column_stack(([w[:, self.feature_time_range_samples[ch], ch] for ch in range(self.n_channels_this_shank)]))
        # Raw PCs by SVD decomposition
        PCs = np.zeros((all_spike_indices.shape[0], 3), dtype=float)
        res = svd(np.cov(w.transpose()))
        PCs[:, :2] = np.dot(w, res[0][:, :2])
        # Spike energy-normalized waveforms
        energy = np.sqrt(np.sum(w ** 2, axis=1)) / float(self.CS.spike_waveform_duration)
        self.DATA.loc[all_spike_indices, 'energy'] = energy
        w = np.divide(w, np.atleast_2d(energy).transpose())
        # Perform SVD decomposition of these waveforms
        res = svd(np.cov(w.transpose()))
        # Multiply the selected PC by the waveforms to obtain a single value for that spike
        PCs[:, 2] = np.dot(w, res[0][:, 0])
        # Store data
        self.DATA.loc[all_spike_indices, ['PC1', 'PC2', 'PC1_norm']] = PCs
        # Make sure that at least 1 feature related to PCs is visible
        plotted_features = [self.OSSA.plotted_features['SpikeSorter']['x']['feature'], self.OSSA.plotted_features['SpikeSorter']['y']['feature']]
        is_any_PC = any(['PC' in i for i in plotted_features])
        if not is_any_PC:
            self.OSSA.plotted_features['SpikeSorter']['x']['feature'] = 'PC1'
            self.OSSA.plotted_features['SpikeSorter']['y']['feature'] = 'PC2'
            self.ax.feature.set_xlabel(translate_feature_to_axis_label('PC1'))
            self.ax.feature.set_ylabel(translate_feature_to_axis_label('PC2'))
        # Update feature plot
        self._update_plot(self.currently_selected_clusters, 'feature', update_data=True, update_limits=True)
        self.show_in_console_and_statusbar('(ok)')

    def callback_feature_time_range_changed(self, which_end, channel_idx):
        if which_end == 'start':
            self.feature_time_range_ms[channel_idx][0] = self.feature_time_range_slider[channel_idx].start()
        elif which_end == 'end':
            self.feature_time_range_ms[channel_idx][1] = self.feature_time_range_slider[channel_idx].end()
        left_end, right_end = self.feature_time_range_ms[channel_idx]
        left_end_sample = np.abs(self.waveform_sample_time - float(left_end)).argmin()
        right_end_sample = np.abs(self.waveform_sample_time - float(right_end)).argmin()
        if right_end_sample == self.waveform_sample_time.shape[0]:
            right_end_sample -= 1
        self.feature_time_range_samples[channel_idx] = np.arange(left_end_sample, right_end_sample + 1)

    def callback_show_help(self):
        self.helperWindow.on_show()
        self.show_in_console_and_statusbar('Showing helper')


    ############################################################################
    # Misc methods
    ############################################################################
    @staticmethod
    def _get_clicked_plot_info(artist):
        """When picking a spike point, extract its info from the label."""
        selected_plot_label = artist.get_label()
        cluster_id = int(selected_plot_label)
        return cluster_id

    def show_in_console_and_statusbar(self, msg):
        LOGGER.info(msg)
        self.menu.statusBar.showMessage(msg)

    def is_waveform_picking_allowed(self, from_plot):
        # No, if the panning tool is on
        if self.currently_panning:
            return False
        # Yes, if the polygon selector is not active
        if not self.currently_selecting_spikes:
            return True
        # If the polygon selector is on, check whether the polygon on the current
        # axes is still active for drawing
        if self.callbacks.polygon_selectors[from_plot].currently_drawing_polygon:
            return False
        else:
            return True

    @staticmethod
    def natural_sort(the_list):
        """Sort the given list in the way that humans expect."""

        def alphanum_key(s):
            return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]
        the_list.sort(key=alphanum_key)
        return the_list


################################################################################
# Implementation of interactive QTableWidget for this GUI
################################################################################
class MultiColumn_QTable(QtWidgets.QTableWidget):
    def __init__(self, data, maximum_height):
        """This class creates a custom QTable which holds some data and creates
        pushbuttons and choice items."""
        super(MultiColumn_QTable, self).__init__()

        # Store the maximum height allowed for long tables
        self.maximum_height = maximum_height

        # Make a stylesheet that creates consistent colors between selection states
        self.setStyleSheet("""QTableWidget:item:selected:active {background:#3399FF; color:white}
                            QTableWidget:item:selected:!active {background:gray; color:white}
                            QTableWidget:item:selected:disabled {background:gray; color:white}
                            QTableWidget:item:selected:!disabled {background:#3399FF; color:white}""")

        # Initialize field names
        self.data = None
        self.n_rows = 0
        self.cluster_type_picker = None
        self.color_picker = None
        self.allowed_interaction = None
        self._columns = ['id', 'n_spikes', 'FR', 'color']

        # Hide row numbers
        self.verticalHeader().setVisible(False)

        # Assign data to the table
        self.cluster_type = data['cluster_type'].values
        self._assign_data(data.loc[:, self._columns])
        # Get default properties
        self._EditTriggers = self.editTriggers()
        self._FocusPolicy = self.focusPolicy()
        self._SelectionMode = self.selectionMode()
        # Start with interaction enabled
        self.enable_interaction()

    def _assign_data(self, data):
        """Reformat data."""
        self.data = data
        self.data.reset_index(drop=True, inplace=True)
        self._get_count_of_rows_and_columns()
        # Make headers list
        self.column_headers = self.keys.tolist()
        format_text = None
        for column in self.column_headers:
            if column != 'color':
                # Add text items
                if column == 'id' or column == 'n_spikes':
                    format_text = lambda x: str('%i' % x)
                elif column == 'FR':
                    format_text = lambda x: str('%.1f' % x)
                for row in range(self.n_rows):
                    item = self.data.loc[row, column]
                    self.data.loc[row, column] = format_text(item)
        self.column_headers += ['type']

        # Make table
        self._make_table()

        # Only entire rows can be selected
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        # Rows can be selected with Ctrl and Shift like one would expect in Windows
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        # Set vertical scrollbars as always visible, but hide horizontal ones
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # Resize table
        self._adjust_table_size()

    def _get_count_of_rows_and_columns(self):
        # Reshape table
        self.keys = self.data.columns
        self.n_columns = len(self.keys) + 1
        self.n_rows = self.data.shape[0]
        self.setRowCount(self.n_rows)
        self.setColumnCount(self.n_columns)

    def _make_table(self):
        self.cluster_type_picker = list(np.empty((int(self.n_rows), ), dtype=object))
        self.color_picker = list(np.empty((int(self.n_rows),), dtype=object))
        # Make table
        for col_idx, col_name in enumerate(self.column_headers):
            if col_name == 'color':
                # Add color picker
                for row_idx in range(self.n_rows):
                    item = self.data.loc[row_idx, col_name]
                    if self.data.loc[row_idx, 'id'] == '0':  # Unsorted spikes
                        newitem = QtWidgets.QLabel("")
                        color_Qt_button(newitem, item)
                        self.setCellWidget(row_idx, col_idx, newitem)
                    else:
                        self.color_picker[row_idx] = QtWidgets.QPushButton("")
                        color_Qt_button(self.color_picker[row_idx], item)
                        self.setCellWidget(row_idx, col_idx, self.color_picker[row_idx])

            elif col_name == 'type':
                for row_idx in range(self.n_rows):
                    if self.data.loc[row_idx, 'id'] == '0':  # Unsorted spikes
                        newitem = QtWidgets.QTableWidgetItem('noise')
                        newitem.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                        self.setItem(row_idx, self.n_columns-1, newitem)
                    else:
                        self.cluster_type_picker[row_idx] = self._make_combo_box(self.cluster_type[row_idx])
                        self.setCellWidget(row_idx, self.n_columns-1, self.cluster_type_picker[row_idx])

            else:
                for row_idx in range(self.n_rows):
                    item = self.data.loc[row_idx, col_name]
                    newitem = QtWidgets.QTableWidgetItem(item)
                    newitem.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                    self.setItem(row_idx, col_idx, newitem)

        # Set the header labels
        self.setHorizontalHeaderLabels(self.column_headers)
        # Resize cell size to content
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def _adjust_table_size(self):
        """Adapt table size to current content"""
        self._getQTableWidgetSize()
        val = np.min((self.maximum_height, self.table_size.height()))
        tableSize = QtCore.QSize(self.table_size.width(), val)
        self.setMinimumSize(tableSize)
        self.setMaximumSize(self.table_size)

    def _getQTableWidgetSize(self):
        """Calculate the right table size for the current content"""
        w = self.verticalHeader().width() + 4 + 20  # +4 seems to be needed
        for i in range(self.columnCount()):
            w += self.columnWidth(i) # seems to include grid line
        h = self.horizontalHeader().height() + 4
        for i in range(self.rowCount()):
            h += self.rowHeight(i)
        self.table_size = QtCore.QSize(w, h)

    def update_table_data(self, data):
        """Update all the data in the table."""
        self.cluster_type = data['cluster_type'].values
        self._assign_data(data.loc[:, self._columns])

    def add_row(self, row_data):
        """Add data to the table"""
        # row_data is an array
        self.n_rows = self.rowCount()
        row_idx = self.n_rows
        self.n_rows += 1
        self.setRowCount(self.n_rows)
        # id
        newitem = QtWidgets.QTableWidgetItem(str('%i' % row_data['id']))
        newitem.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        self.setItem(row_idx, 0, newitem)
        # n_spikes
        newitem = QtWidgets.QTableWidgetItem(str('%i' % row_data['n_spikes']))
        newitem.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        self.setItem(row_idx, 1, newitem)
        # FR
        newitem = QtWidgets.QTableWidgetItem(str('%.1f' % row_data['FR']))
        newitem.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        self.setItem(row_idx, 2, newitem)
        # Color picker
        self.color_picker.append(QtWidgets.QPushButton(""))
        color_Qt_button(self.color_picker[-1], row_data['color'])
        self.setCellWidget(row_idx, 3, self.color_picker[-1])
        # cluster type
        self.cluster_type_picker.append(self._make_combo_box(default_combo_box_value=2))
        self.setCellWidget(row_idx, 4, self.cluster_type_picker[-1])
        # Resize cell size to content
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        # Resize table
        self._adjust_table_size()

    def update_row(self, row_idx, row_data):
        """Update specific row."""
        self.item(row_idx, 1).setText(str('%i' % row_data['n_spikes']))
        self.item(row_idx, 2).setText(str('%.1f' % row_data['FR']))
        self._adjust_table_size()

    def remove_row(self, row_idx):
        """Remove specific row"""
        self.removeRow(row_idx)
        self.color_picker.pop(row_idx)
        self.cluster_type_picker.pop(row_idx)
        self.n_rows = self.rowCount()

    def disable_interaction(self):
        """Make table not interactive."""
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        for row in range(1, self.n_rows):
            self.color_picker[row].setEnabled(False)
            self.cluster_type_picker[row].setEnabled(False)
        self.allowed_interaction = False

    def enable_interaction(self):
        """Make table interactive."""
        self.setEditTriggers(self._EditTriggers)
        self.setFocusPolicy(self._FocusPolicy)
        self.setSelectionMode(self._SelectionMode)
        for row in range(1, self.n_rows):
            self.color_picker[row].setEnabled(True)
            self.cluster_type_picker[row].setEnabled(True)
        self.allowed_interaction = True

    @staticmethod
    def get_row_number(items):
        """Return the row index of each item in <items>."""
        return np.array([i.row() for i in items])

    def get_row_containing(self, value, column=0, dtype=int):
        """Return the row index of the column containing <value>."""
        output_row = None
        for row in range(self.n_rows):
            this_value = dtype(str(self.item(row, column).text()))
            if this_value == value:
                output_row = row
                break
        if output_row is None:
            output_row = self.n_rows
        return output_row

    @staticmethod
    def return_from_column(column, items, data_type):
        """Return items on columns."""
        idx = np.where(np.array([i.column() for i in items]) == column)[0]
        return np.array([data_type(items[i].text()) for i in idx])

    @staticmethod
    def _make_combo_box(default_combo_box_value):
        """Make a combo-box, i.e., a drop-down menu."""
        combo_box_options = ['noise', 'SU', 'MU']
        combo = QtWidgets.QComboBox()
        for t in combo_box_options:
            combo.addItem(t)
        # Convert string value to numeric
        if isinstance(default_combo_box_value, str):
            default_combo_box_value = [i.lower()[0] for i in combo_box_options].index(default_combo_box_value)
        combo.setCurrentIndex(default_combo_box_value)
        return combo
