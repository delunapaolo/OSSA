"""Make dictionaries holding all the default values for the GUIs."""

# Make sure we use Qt5
import matplotlib
matplotlib.use('Qt5Agg')
# Imports to generate icons and fonts
from PyQt5 import QtGui
from Code.Workflows.Spike_sorting_workflow.OSSA.icons import *


# This dictionary contains the default values for analyses.
analysis = dict()
analysis['history_size'] = 5  # Number of times the 'undo' function can be called
# Auto- and Cross-correlograms
analysis['ACG_bin_size'] = 1  # ms
analysis['ACG_max_lag'] = 50 # ms
# Parameters for ISI distributions
analysis['ISI_bin_size'] = 0.5  # ms
analysis['ISI_threshold'] = 2  # ms


# This dictionary contains the columns in the .spikes.hdf5 file that will be used
# for the plots in the GUI. The x-axis of the stability plot cannot be changed.
plotted_features = dict({'NoiseRemoval':dict({'x':dict(), 'y':dict()}), 'SpikeSorter':dict({'x':dict(), 'y':dict()})})
plotted_features['NoiseRemoval']['y']['stability'] = 'peak_amplitude_SD'
plotted_features['NoiseRemoval']['x']['feature'] = 'trough_amplitude_SD'
plotted_features['NoiseRemoval']['y']['feature'] = 'peak_amplitude_SD'
plotted_features['SpikeSorter']['y']['stability'] = 'peak_amplitude_SD'
plotted_features['SpikeSorter']['x']['feature'] = 'PC1_norm'
plotted_features['SpikeSorter']['y']['feature'] = 'energy'


# This dictionary contains the keyboard shortcuts used in the GUI. Three are
# hardcoded: they are Ctrl+S for Save, Ctrl+Q for Quit, and Ctrl+R for reload.
# The workflow does not check whether the same shortcut is already in use. Please
# use at your own discretion. More info on how shortcuts are internally translated
# can be found here: http://pyqt.sourceforge.net/Docs/PyQt4/qkeysequence.html
keyboard_shortcuts = dict({'NoiseRemoval':dict(), 'SpikeSorter':dict(), 'ChannelSelector':dict()})
keyboard_shortcuts['save_and_close'] = 'Ctrl+S'
keyboard_shortcuts['discard_and_close'] = 'Ctrl+W'
keyboard_shortcuts['reload'] = 'Ctrl+R'
keyboard_shortcuts['pan_zoom'] = 'z'
keyboard_shortcuts['reset_zoom'] = 'h'
keyboard_shortcuts['undo'] = 'Ctrl+Z'
keyboard_shortcuts['redo'] = 'Ctrl+Y'

keyboard_shortcuts['ChannelSelector']['quit_workflow'] = 'Ctrl+Q'

keyboard_shortcuts['NoiseRemoval']['keep_spikes'] = 'c'
keyboard_shortcuts['NoiseRemoval']['discard_spikes'] = 'x'
keyboard_shortcuts['NoiseRemoval']['toggle_suggestions'] = 's'

keyboard_shortcuts['SpikeSorter']['add_cluster'] = 'c'
keyboard_shortcuts['SpikeSorter']['edit_cluster'] = 'x'
keyboard_shortcuts['SpikeSorter']['delete_cluster'] = 'Delete'
keyboard_shortcuts['SpikeSorter']['merge_clusters'] = 'v'


################################################################################
# This dictionary contains the default values for graphical elements.
default = dict()
# Colors and sizes
default['color_black'] = (0., 0., 0.)
default['color_white'] = (1., 1., 1.)
default['color_red'] = (1., .39, .28)
default['color_dark_green'] = (0., .5, 0.)
default['color_light_green'] = (.5, 1., 0.)
default['color_light_gray'] = (.92, .92, .95)
default['color_dark_gray'] = (.7, .7, .7)
default['color_orange'] = (1., .65, 0.)
default['axes_color'] = (.96, .96, .96)
default['line_width_thin'] = 1.5
default['line_width_thick'] = 2
default['last_selected_point_size'] = 13
default['scatter_size'] = 9
# Icons
default['icon_polygon'] = icon_polygon
default['icon_magnifying_glass'] = icon_magnifying_glass
default['icon_visible'] = icon_eye
default['icon_invisible'] = icon_eye_slash
default['icon_add'] = icon_plus
default['icon_delete'] = icon_delete
default['icon_edit'] = icon_pencil
default['icon_merge'] = icon_converging_arrows
default['icon_home'] = icon_home
default['icon_clean'] = icon_clean
# Size of QWidgets in layouts
default['widget_minimum_size'] = 150
# Size of a printed plot for CCR between all channels, in inches
default['printed_plot_inches'] = 3.5

# Default values for the ChannelSelector GUI
default['ChannelSelector'] = dict()
default['ChannelSelector']['font'] = QtGui.QFont()
default['ChannelSelector']['font'].setFamily('Lucida')
default['ChannelSelector']['font'].setPointSize(12)
default['ChannelSelector']['channel_picker_tolerance'] = 15  # pixels around a data-point to trigger the callback
default['ChannelSelector']['button_width'] = 0.1
default['ChannelSelector']['button_height'] = 0.1
default['ChannelSelector']['color_channel_active'] = default['color_black']
default['ChannelSelector']['color_channel_inactive'] = default['color_dark_gray']
default['ChannelSelector']['color_channel_highlight'] = (1., 1., 0., .8)
default['ChannelSelector']['color_channel_keep'] =  default['color_orange']
default['ChannelSelector']['color_channel_discard'] = default['color_red']
default['ChannelSelector']['color_channel_sorted'] = default['color_dark_green']

# Default values for the NoiseRemoval GUI
default['NoiseRemoval'] = dict()
default['NoiseRemoval']['waveform_plot_style'] = 'horizontal'
default['NoiseRemoval']['color_good_spikes'] = default['color_black']
default['NoiseRemoval']['color_bad_spikes'] = default['color_dark_gray']
default['NoiseRemoval']['size_bad_spikes'] = 7
default['NoiseRemoval']['stability_plot_textbox_props'] = dict(boxstyle='square', fc='w', ec='0.5', alpha=0.9)
default['NoiseRemoval']['channel_button_color_neutral'] = default['color_light_gray']
default['NoiseRemoval']['channel_button_color_keep'] = default['color_light_green']
default['NoiseRemoval']['channel_button_color_discard'] = default['color_red']
default['NoiseRemoval']['selector_color_keep'] = default['color_light_green']
default['NoiseRemoval']['selector_color_discard'] = default['color_red']

# Default values for the SpikeSorter GUI
default['SpikeSorter'] = dict()
default['SpikeSorter']['color_unsorted_spikes'] = default['color_dark_gray']
default['SpikeSorter']['button_color_neutral'] = default['color_light_gray']
default['SpikeSorter']['button_color_add'] = default['color_light_green']
default['SpikeSorter']['button_color_delete'] = default['color_red']
default['SpikeSorter']['stability_plot_textbox_props'] = dict(boxstyle='square', fc='w', ec='0.5', alpha=0.9)
default['SpikeSorter']['selector_color_add'] = default['color_light_green']
default['SpikeSorter']['selector_color_discard'] = default['color_red']

