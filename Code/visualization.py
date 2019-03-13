"""This module contains Qt-related re-implementations and other visualization /
plotting functions for OSSA GUIs."""

# Graphical
from PyQt5 import QtGui, QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import _SelectorWidget, ToolHandles
from matplotlib.lines import Line2D
from matplotlib.path import Path
import seaborn as sns

# Numerical computation
import numpy as np

# Local repository
from Code.GUIs.GUI_utilities import NoCloseWindow
from Code.Workflows.Spike_sorting_workflow.OSSA.icons import OSSA_taskbar_logo


################################################################################
# Qt-related
################################################################################
class OSSA_window(NoCloseWindow):
    def __init__(self, title=''):
        super(OSSA_window, self).__init__()

        # Set window title
        if title != '':
            self.setWindowTitle(title)
        else:
            self.setWindowTitle('OSSA')
        # Set window taskbar icon
        self.setWindowIcon(QtGui.QIcon(QtGui.QPixmap(OSSA_taskbar_logo)))

class OSSAHelpDialog(QtWidgets.QMessageBox):
    """We implement a custom message box to show the helper dialog with the
    keyboard shortcuts."""

    def __init__(self, shortcuts):
        # Initialize super-class
        super(OSSAHelpDialog, self).__init__()

        # Add 2 tabs between shortcuts and their description
        separator = '\t\t'

        # Prepare text by separating lines and removing leading and trailing spaces
        lines = shortcuts.split('\n')
        lines = [i.strip() for i in lines]
        # Remove empty lines at beginning and end
        if lines[0] == '':
            lines.pop(0)
        if lines[-1] == '':
            lines.pop(-1)
        # Get length of each string representing a shortcut
        txt = [i.split('\t') for i in lines]
        kb_shortcuts = list()
        for l in txt:
            if len(l) == 2:
                kb_shortcuts.append(l[0])
        # Calculate how many blank characters should be added at the end of each
        # string to have equal-width columns
        kb_shortcuts_len = np.array([len(i) for i in kb_shortcuts], dtype=int)
        padding = kb_shortcuts_len.max() - kb_shortcuts_len

        # Loop through each line and construct text lines
        msg = 'Esc' + ' ' * (kb_shortcuts_len.max() - 3) + separator + 'Close help\n\n'
        idx = 0
        for l in txt:
            if len(l) == 2:
                msg += l[0] + ' ' * padding[idx] + separator + l[1]
                idx += 1
            elif len(l) == 1 and l[0] != '':
                msg += l[0]
            msg += '\n'
        # Remove trailing empty line
        msg = msg[:-1]

        # Set text
        self.setText(msg)
        self.setTextFormat(QtCore.Qt.PlainText)
        self.setFont(QtGui.QFont('Lucida', 14))
        self.setWindowTitle('Help - Keyboard shortcuts')

        # Make window non-interrupting but without buttons
        self.setModal(False)
        self.setSizeGripEnabled(False)
        self.setStandardButtons(QtWidgets.QMessageBox.NoButton)

        # Add a keyboard shortcut to close the current window
        self.Key_Escape = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        self.Key_Escape.setAutoRepeat(False)
        self.Key_Escape.activated.connect(self.callback_exit)

    def on_show(self):
        self.show()

    def callback_exit(self):
        self.hide()

class OSSAFigureCanvas(FigureCanvas):
    """We re-implement the class FigureCanvas to create a custom signal to emit
    when the widget is resized."""
    resized = QtCore.pyqtSignal()
    def __init__(self, figure, parent=None):
        super(OSSAFigureCanvas, self).__init__(figure)
        self.setParent(parent)
        # This is needed to allow callbacks from embedded matplotlib to access pyQt's event handler
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFocus()

    def resizeEvent(self, event):
        """Called whenever the window is resized."""
        self.resized.emit()
        return super(OSSAFigureCanvas, self).resizeEvent(event)

def make_QSplitter(orientation):
    """Make a QSplitter object with default properties."""
    if orientation == 'hor':
        ori = QtCore.Qt.Horizontal
    else:
        ori = QtCore.Qt.Vertical
    s = QtWidgets.QSplitter()
    s.setOrientation(ori)
    s.setChildrenCollapsible(False)
    s.setHandleWidth(7)
    s.setOpaqueResize(True)
    s.setStyleSheet('QSplitter::handle {border: 1px dashed #76797C;}'
                    'QSplitter::handle:pressed {background-color: #787876; border: 1px solid #76797C;}')
    return s

def color_Qt_button(button, color):
    """Assign a color to a button.
    :param button: [object] A pyQt button instance.
    :param color: [iterable with 3 items] Colors in [0-255] range.
    """
    rgb = ','.join(['%i' % (i*255.) for i in color])
    button.setStyleSheet('background-color:rgb(%s)' % rgb)


################################################################################
# QRangeSlider
################################################################################
class Ui_Form(object):
    """default range slider form"""
    def __init__(self):
        self.gridLayout = None
        self._splitter = None
        self._head = None
        self._handle = None
        self._tail = None


    def setupUi(self, Form):
        DEFAULT_CSS = """
        QRangeSlider * {border: 0px; padding: 0px}
        QRangeSlider #Head {background: #F5F5F5}
        QRangeSlider #Span {background: #95A5A6}
        QRangeSlider #Span:active {background: #F4D03F}
        QRangeSlider #Tail {background: #F5F5F5}
        QRangeSlider > QSplitter::handle {border: 1px dashed #76797C}
        QRangeSlider > QSplitter::handle:vertical {height: 4px}
        QRangeSlider > QSplitter::handle:pressed {background-color: #787876}
        """

        try:
            _fromUtf8 = QtCore.QString.fromUtf8
        except AttributeError:
            _fromUtf8 = lambda s: s

        Form.setObjectName(_fromUtf8('QRangeSlider'))
        Form.resize(300, 30)
        Form.setStyleSheet(_fromUtf8(DEFAULT_CSS))
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(_fromUtf8('gridLayout'))
        self._splitter = QtWidgets.QSplitter(Form)
        self._splitter.setMinimumSize(QtCore.QSize(0, 0))
        self._splitter.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self._splitter.setOrientation(QtCore.Qt.Horizontal)
        self._splitter.setObjectName(_fromUtf8('splitter'))
        self._splitter.setHandleWidth(7)
        self._head = QtWidgets.QGroupBox(self._splitter)
        self._head.setTitle(_fromUtf8(''))
        self._head.setObjectName(_fromUtf8('Head'))
        self._handle = QtWidgets.QGroupBox(self._splitter)
        self._handle.setTitle(_fromUtf8(''))
        self._handle.setObjectName(_fromUtf8('Span'))
        self._tail = QtWidgets.QGroupBox(self._splitter)
        self._tail.setTitle(_fromUtf8(''))
        self._tail.setObjectName(_fromUtf8('Tail'))
        self.gridLayout.addWidget(self._splitter, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    @staticmethod
    def retranslateUi(Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate('QRangeSlider', 'QRangeSlider', None))

class Element(QtWidgets.QGroupBox):
    def __init__(self, parent, main, str_format='%i'):
        super(Element, self).__init__(parent)
        self.main = main
        self.format = str_format

    def setStyleSheet(self, style):
        """redirect style to parent groupbox"""
        self.parent().setStyleSheet(style)

    def textColor(self):
        """text paint color"""
        return getattr(self, '__textColor', QtGui.QColor(125, 125, 125))

    def setTextColor(self, color):
        """set the text paint color"""
        if type(color) == tuple and len(color) == 3:
            color = QtGui.QColor(color[0], color[1], color[2])
        elif type(color) == int:
            color = QtGui.QColor(color, color, color)
        setattr(self, '__textColor', color)

    def paintEvent(self, event):
        """overrides paint event to handle text"""
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.main.drawValues():
            self.drawText(event, qp)
        qp.end()

class Head(Element):
    """area before the handle"""
    def __init__(self, parent, main, str_format='%i'):
        super(Head, self).__init__(parent, main, str_format)

    def drawText(self, event, qp):
        event.ignore()
        # qp.setPen(self.textColor())
        # qp.setFont(QtGui.QFont('Arial', 10))
        # qp.drawText(event.rect(), QtCore.Qt.AlignLeft, str(self.format % self.main.min()))

class Tail(Element):
    """area after the handle"""
    def __init__(self, parent, main, str_format='%i'):
        super(Tail, self).__init__(parent, main, str_format)

    def drawText(self, event, qp):
        event.ignore()
        # qp.setPen(self.textColor())
        # qp.setFont(QtGui.QFont('Arial', 10))
        # qp.drawText(event.rect(), QtCore.Qt.AlignRight, str(self.format % self.main.max()))

class Handle(Element):
    """handle area"""
    def __init__(self, parent, main, str_format='%i'):
        super(Handle, self).__init__(parent, main, str_format)

    def drawText(self, event, qp):
        qp.setPen(self.textColor())
        qp.setFont(QtGui.QFont('Arial', 10))
        qp.drawText(event.rect(), QtCore.Qt.AlignLeft, str(self.format % self.main.start()))
        qp.drawText(event.rect(), QtCore.Qt.AlignRight, str(self.format % self.main.end()))

    def mouseMoveEvent(self, event):
        event.ignore()
        return
        # event.accept()
        # mx = event.globalX()
        # _mx = getattr(self, '__mx', None)
        #
        # if not _mx:
        #     setattr(self, '__mx', mx)
        #     dx = 0
        # else:
        #     dx = mx - _mx
        #
        # setattr(self, '__mx', mx)
        #
        # if dx == 0:
        #     event.ignore()
        #     return
        # elif dx > 0:
        #     dx = 1
        # elif dx < 0:
        #     dx = -1
        #
        # s = self.main.start() + dx
        # e = self.main.end() + dx
        # if s >= self.main.min() and e <= self.main.max():
        #     self.main.setRange(s, e)

class QRangeSlider(QtWidgets.QWidget, Ui_Form):
    """
    The QRangeSlider class implements a horizontal range slider widget.
    """
    # signals
    startValueChanged = QtCore.pyqtSignal(int)
    endValueChanged = QtCore.pyqtSignal(int)

    # define splitter indices
    _SPLIT_START = 1
    _SPLIT_END = 2

    def __init__(self, parent=None, min_value=0, max_value=100, n_decimals=0):
        """Create a new QRangeSlider instance.

            :param parent: QWidget parent
            :return: New QRangeSlider instance.

        """
        super(QRangeSlider, self).__init__(parent)
        self.n_decimals = n_decimals
        self.setupUi(self)
        self.setMouseTracking(False)

        # Add parameters to interpret step
        if n_decimals == 0:
            self.valuesType = int
            str_format = '%i'
        else:
            self.valuesType = float
            str_format = '%%.%if' % self.n_decimals

        self._splitter.splitterMoved.connect(self._handleMoveSplitter)

        # head layout
        self._head_layout = QtWidgets.QHBoxLayout()
        self._head_layout.setSpacing(0)
        self._head_layout.setContentsMargins(0, 0, 0, 0,)
        self._head.setLayout(self._head_layout)
        self.head = Head(self._head, main=self, str_format=str_format)
        self._head_layout.addWidget(self.head)
        self.head.setTextColor((0, 0, 0))

        # handle layout
        self._handle_layout = QtWidgets.QHBoxLayout()
        self._handle_layout.setSpacing(0)
        self._handle_layout.setContentsMargins(0, 0, 0, 0)
        self._handle.setLayout(self._handle_layout)
        self.handle = Handle(self._handle, main=self, str_format=str_format)
        self.handle.setTextColor((255, 0, 0))
        self._handle_layout.addWidget(self.handle)

        # tail layout
        self._tail_layout = QtWidgets.QHBoxLayout()
        self._tail_layout.setSpacing(0)
        self._tail_layout.setContentsMargins(0, 0, 0, 0)
        self._tail.setLayout(self._tail_layout)
        self.tail = Tail(self._tail, main=self, str_format=str_format)
        self._tail_layout.addWidget(self.tail)
        self.tail.setTextColor((0, 0, 0))

        # defaults
        self.__setMin(min_value)
        self.__setMax(max_value)
        self.setStart(min_value)
        self.setEnd(max_value)
        self.setDrawValues(True)


    ############################################################################
    # Getters
    ############################################################################
    def min(self):
        """:return: minimum value"""
        return getattr(self, '__min', None)

    def max(self):
        """:return: maximum value"""
        return getattr(self, '__max', None)

    def start(self):
        """:return: range slider start value"""
        return getattr(self, '__start', None)

    def end(self):
        """:return: range slider end value"""
        return getattr(self, '__end', None)

    def getRange(self):
        """:return: the start and end values as a tuple"""
        return self.start(), self.end()

    def drawValues(self):
        """:return: True if slider values will be drawn"""
        return getattr(self, '__drawValues', None)


    ############################################################################
    # Setters
    ############################################################################
    def __setMin(self, value):
        """sets minimum value"""
        value = self.valuesType(value)
        setattr(self, '__min', value)

    def __setMax(self, value):
        """sets maximum value"""
        value = self.valuesType(value)
        setattr(self, '__max', value)

    def _setStart(self, value):
        """stores the start value only"""
        value = self.valuesType(value)
        setattr(self, '__start', value)
        self.startValueChanged.emit(value)

    def setStart(self, value):
        """sets the range slider start value"""
        v = self._valueToPos(value)
        self._splitter.splitterMoved.disconnect()
        self._splitter.moveSplitter(v, self._SPLIT_START)
        self._splitter.splitterMoved.connect(self._handleMoveSplitter)
        self._setStart(value)

    def _setEnd(self, value):
        """stores the end value only"""
        value = self.valuesType(value)
        setattr(self, '__end', value)
        self.endValueChanged.emit(value)

    def setEnd(self, value):
        """set the range slider end value"""
        v = self._valueToPos(value)
        self._splitter.splitterMoved.disconnect()
        self._splitter.moveSplitter(v, self._SPLIT_END)
        self._splitter.splitterMoved.connect(self._handleMoveSplitter)
        self._setEnd(value)

    def setRange(self, start, end):
        """set the start and end values"""
        self.setStart(self.valuesType(start))
        self.setEnd(self.valuesType(end))

    def setDrawValues(self, draw):
        """sets draw values boolean to draw slider values"""
        assert type(draw) is bool
        setattr(self, '__drawValues', draw)


    ############################################################################
    # Interaction
    ############################################################################
    def keyPressEvent(self, event):
        """overrides key press event to move range left and right"""
        event.ignore()

    def setBackgroundStyle(self, style):
        """sets background style"""
        self._tail.setStyleSheet(style)
        self._head.setStyleSheet(style)

    def setSpanStyle(self, style):
        """sets range span handle style"""
        self._handle.setStyleSheet(style)

    def _handleMoveSplitter(self, xpos, index):
        """private method for handling moving splitter handles"""
        def _lockWidth(widget):
            width = widget.size().width()
            widget.setMinimumWidth(width)
            widget.setMaximumWidth(width)

        def _unlockWidth(widget):
            widget.setMinimumWidth(0)
            widget.setMaximumWidth(16777215)

        # Get actual value
        value = self._posToValue(xpos)

        if index == self._SPLIT_START:
            _lockWidth(self._tail)
            if value >= self.end():
                return
            self._setStart(value)

        elif index == self._SPLIT_END:
            _lockWidth(self._head)
            if value <= self.start():
                return

            # Re-adjust value when it's by the end
            hw = self._splitter.handleWidth()
            value = self.min() + (self.max() - self.min()) / self.width() * (xpos + hw + 2)
            self._setEnd(value)

        _unlockWidth(self._tail)
        _unlockWidth(self._head)
        _unlockWidth(self._handle)


    ############################################################################
    # Helpers
    ############################################################################
    def _valueToPos(self, value):
        """converts slider value to local pixel x coord"""
        xpos = self.valuesType(self.scale(value, (self.min(), self.max()), (0, self.width())))
        return xpos

    def _posToValue(self, xpos):
        """converts local pixel x coord to slider value"""
        value = self.valuesType(self.scale(xpos, (0, self.width()), (self.min(), self.max())))
        return value

    def scale(self, val, src, dst):
        """
        Scale the given value from the scale of src to the scale of dst.
        """
        value = np.round(((val - src[0]) / float(src[1] - src[0])) * (dst[1] - dst[0]) + dst[0], decimals=self.n_decimals)
        return value


################################################################################
# matplotlib-related
################################################################################
def translate_feature_to_axis_label(feature):
    # This function makes sure that given a certain feature name, the axes labels
    # mark what it is plotted
    if feature == 'timestamp':
        return 'time (min)'
    elif feature == 'template_amplitude':
        return 'template matching amplitude'
    elif feature == 'peak_amplitude':
        return 'peak amplitude (mV)'
    elif feature == 'peak_amplitude_SD':
        return 'peak amplitude (MAD)'
    elif feature == 'PC1':
        return 'PC1'
    elif feature == 'PC1_norm':
        return 'energy-normalized PC1'
    elif feature == 'PC2':
        return 'PC2'
    elif feature == 'energy':
        return 'energy'
    elif feature == 'peak_to_trough_usec':
        return 'peak-to-trough (usec)'
    elif feature == 'peak_to_trough_mV':
        return 'peak-to-trough (mV)'
    elif feature == 'trough_amplitude':
        return 'trough amplitude (mV)'
    elif feature == 'trough_amplitude_SD':
        return 'trough amplitude (MAD)'
    else:
        msg = 'ERROR: Unknown feature \'%s\'' % feature
        raise Exception(msg)

def translate_feature_list_to_tooltip(feature_list):
    # This function makes sure that given a certain feature name, the info shown
    # in the tooltip has the right name
    if isinstance(feature_list, str):
        feature_list = [feature_list]
    output_string = list()
    for feature in feature_list:
        if feature == 'timestamp' or feature == 'time':
            output_string += ['time: %.3fs']
        elif feature == 'template_amplitude':
            output_string += ['templ. ampl.: %.2f']
        elif feature == 'peak_amplitude':
            output_string += ['peak: %.2fmV']
        elif feature == 'peak_amplitude_SD':
            output_string += ['peak: %.2f']
        elif feature == 'PC1':
            output_string += ['PC1: %.2f']
        elif feature == 'PC1_norm':
            output_string += ['norm. PC1: %.2f']
        elif feature == 'PC2':
            output_string += ['PC2: %.2f']
        elif feature == 'energy':
            output_string += ['energy: %.2f']
        elif feature == 'peak_to_trough_usec':
            output_string += ['peak-to-trough: %.2fus']
        elif feature == 'peak_to_trough_mV':
            output_string += ['peak-to-trough: %.2fmV']
        elif feature == 'trough_amplitude':
            output_string += ['trough: %.2fmV']
        elif feature == 'trough_amplitude_SD':
            output_string += ['trough: %.2f']
        else:
            msg = 'ERROR: Unknown feature to plot'
            raise Exception(msg)
    return ', '.join(output_string)

def set_figure_style(axes_color, seaborn_style='darkgrid'):
    style_dict = dict({'figure.facecolor': 'white',
                       'axes.facecolor': axes_color,
                       'xtick.major.size': 0,
                       'ytick.major.size': 0,
                       'grid.color': [i*0.885 for i in axes_color],
                       'axes.labelcolor': (.5, .5, .5),
                       'figure.max_open_warning': 1e5})
    sns.set_style(seaborn_style, style_dict)

    return style_dict


################################################################################
# matplotlib's data-point selector
################################################################################
class SelectFromCollection(object):
    """Select indices from a matplotlib collection using 'PolygonSelector'.
    Selected indices are saved in the 'ind' attribute.

    :param ax: Axes to interact with.
    :param x: x-coordinates of points to interact with
    :param y: y-coordinates of points to interact with
    :param points_id: array of ids of all points
    :param color: color of polygon selector. Default is black.
    :param on_close_function: function handle to run when user presses the Enter
        key. By default, it disconnects itself.
    """

    def __init__(self, ax, x, y, points_id=None, color='k', on_close_function=None):
        """Initialize attributes."""
        self.ax = ax
        self.xys = np.vstack((x, y)).transpose()
        if points_id is None:
            n_points = x.shape[0]
            points_id = np.zeros((n_points, 2), dtype=int)
            points_id[:, 1] = np.arange(n_points)
        self.points_id = points_id
        self._color = color
        self.on_close_function = on_close_function

        # Pre-allocate fields
        self.polygon_selector = None
        self.polygon_lines = list()
        self.currently_drawing_polygon = False
        self.polygon_completed = False
        self.all_points_are_selected = False
        self.ind = np.empty((0, ), dtype=int)
        self.callback_handles = np.empty((3, ), dtype=object)
        self._control_key_is_pressed = False
        self._last_key_pressed = ''
        self.exit_code = 1  # 'normal' exit

        # Instantiate the PolygonSelector class
        self._connect_callbacks()
        self._add_polygon()

    def _add_polygon(self, starting_point=None):
        self.currently_drawing_polygon = True
        self.polygon_selector = PolygonSelector(self.ax, self.when_done, useblit=True, lineprops=dict({'color': self._color, 'linewidth': 2, 'alpha': 1.}), markerprops=dict({'mec': self._color, 'mfc': self._color, 'alpha': 1.}), mark_point=starting_point)
        # Redraw canvas
        self.ax.figure.canvas.draw()

    def _remove_polygon(self):
        # Deactivate and disconnect the current polygon selector
        if self.polygon_selector.get_active():
            self.polygon_selector.set_active(False)
            self.polygon_selector.disconnect_events()
            # Redraw canvas
            self.ax.figure.canvas.draw()

    def _connect_callbacks(self):
        """Associate callback functions to figures and store handle."""
        self.callback_handles[0] = self.ax.figure.canvas.mpl_connect('key_press_event', self.callback_key_press)
        self.callback_handles[1] = self.ax.figure.canvas.mpl_connect('key_release_event', self.callback_key_release)
        self.callback_handles[2] = self.ax.figure.canvas.mpl_connect('button_press_event', self.callback_Ctrl_click)

    def _disconnect_callbacks(self):
        """Disassociate callback functions to figures and delete handle."""
        self.ax.figure.canvas.mpl_disconnect(self.callback_handles[0])
        self.ax.figure.canvas.mpl_disconnect(self.callback_handles[1])
        self.ax.figure.canvas.mpl_disconnect(self.callback_handles[2])
        self.callback_handles = np.empty((3, ), dtype=object)

    def callback_key_press(self, event):
        """In response to key press."""
        # If Ctrl key is pressed, mark it
        if event.key == 'control':
            self._control_key_is_pressed = True

        # If Enter is pressed, accept and close
        elif event.key == 'enter':
            # Disconnect connector
            self.disconnect()
            # Call user's function
            if self.on_close_function is not None:
               self.on_close_function()

        # If Escape key is pressed, restart the selector
        elif event.key == 'escape':
            # If the user pressed Esc twice, reset data and close
            if self._last_key_pressed == 'escape':
                self.exit_code = 0  # 'error' exit
                self.currently_drawing_polygon = False
                self.polygon_completed = False
                self.all_points_are_selected = False
                self.ind = list()
                self.disconnect()
                # Call user's function
                if self.on_close_function is not None:
                   self.on_close_function()
            else:
                self._remove_polygon()
                self._add_polygon()

        # If Ctrl+A is pressed, select all points and stop selector
        elif event.key == 'ctrl+a':
            self.all_points_are_selected = True
            if self.xys.shape[0] == 0:
                x_min = 0
                x_max = 1
                y_min = 0
                y_max = 1
            else:
                padding_percent_range = 0.01  # 1%
                # Compute vertices of the area that contains all the points. Add some
                # padding to make sure all points will be selected
                x_min = self.xys[:, 0].min()
                x_max = self.xys[:, 0].max()
                y_min = self.xys[:, 1].min()
                y_max = self.xys[:, 1].max()
                # Add 1% of the range
                x_padding = (x_max - x_min) * padding_percent_range
                y_padding = (y_max - y_min) * padding_percent_range
                x_min -= x_padding
                x_max += x_padding
                y_min -= y_padding
                y_max += y_padding
            # Vertices of the rectangular area go counter-clockwise from top-left
            verts = [(x_min, y_max), (x_min, y_min), (x_max, y_min), (x_max, y_max)]
            # Finish operation
            self.when_done(verts)

        # Store the last key pressed
        self._last_key_pressed = event.key

    def callback_key_release(self, event):
        """When a key is released."""
        # If the Ctrl key is released, take note of it.
        if event.key == 'control':
            self._control_key_is_pressed = False

    def callback_Ctrl_click(self, event):
        """When the combination control + mouse click is pressed."""
        # This function starts a new polygon selector if the previous one was closed
        if self._control_key_is_pressed and not self.currently_drawing_polygon and not self.all_points_are_selected:
            # Reset the button state to avoid multiple clicks without releasing
            # the Ctrl button
            self._control_key_is_pressed = False
            # Get current x and y data
            starting_point = self.ax.transData.inverted().transform((event.x,  event.y))
            # Add new polygon selector
            self._add_polygon(starting_point=starting_point)

    def when_done(self, verts):
        """This function is called when the polygon is closed."""
        # Extract vertices of the polygon, the indices of the points included
        # in it; Use these indices to find the coordinates of the selected points.
        path = Path(verts)
        ind = np.nonzero(path.contains_points(self.xys))[0]
        self.ind = np.union1d(self.ind, ind)
        # Change flags
        self.currently_drawing_polygon = False
        self.polygon_completed = True
        # Redraw polygon as static line
        if self.all_points_are_selected:
            verts = np.array(verts)
            verts = np.vstack((verts, verts[0, :]))
            x = verts[:, 0]
            y = verts[:, 1]
        else:
            # Plot the vertices of the last polygon before it gets disconnected
            x = self.polygon_selector.line.get_xdata()
            y = self.polygon_selector.line.get_ydata()
        self.polygon_lines.append(self.ax.plot(x, y, color=self._color, scalex=False, scaley=False, zorder=10000, linewidth=1.8))
        # Delete polygon
        self._remove_polygon()

    def disconnect(self):
        """Called when the PolygonSelector gets disconnected."""
        # Disconnect callbacks
        self._disconnect_callbacks()
        # Disconnect the last selector
        self._remove_polygon()
        # Delete all the lines
        for p in range(len(self.polygon_lines)):
            l = self.polygon_lines.pop()
            self.ax.lines.remove(l[0])
        # Redraw canvas
        self.ax.figure.canvas.draw()

class PolygonSelector(_SelectorWidget):
    """Select a polygon region of an axes.
    Place vertices with each mouse click, and make the selection by completing
    the polygon (clicking on the first vertex). Hold the *ctrl* key and click
    and drag a vertex to reposition it (the *ctrl* key is not necessary if the
    polygon has already been completed). Hold the *shift* key and click and
    drag anywhere in the axes to move all vertices. Press the *esc* key to
    start a new polygon.
    For the selector to remain responsive you must keep a reference to
    it.
    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        The parent axes for the widget.
    onselect : function
        When a polygon is completed or modified after completion,
        the `onselect` function is called and passed a list of the vertices as
        ``(xdata, ydata)`` tuples.
    useblit : bool, optional
    lineprops : dict, optional
        The line for the sides of the polygon is drawn with the properties
        given by `lineprops`. The default is ``dict(color='k', linestyle='-',
        linewidth=2, alpha=0.5)``.
    markerprops : dict, optional
        The markers for the vertices of the polygon are drawn with the
        properties given by `markerprops`. The default is ``dict(marker='o',
        markersize=7, mec='k', mfc='k', alpha=0.5)``.
    vertex_select_radius : float, optional
        A vertex is selected (to complete the polygon or to move a vertex)
        if the mouse click is within `vertex_select_radius` pixels of the
        vertex. The default radius is 15 pixels.
    See Also
    --------
    :ref:`sphx_glr_gallery_widgets_polygon_selector_demo.py`
    """

    def __init__(self, ax, onselect, useblit=False, lineprops=None, markerprops=None, vertex_select_radius=15, mark_point=None):
        # The state modifiers 'move', 'square', and 'center' are expected by
        # _SelectorWidget but are not supported by PolygonSelector
        # Note: could not use the existing 'move' state modifier in-place of
        # 'move_all' because _SelectorWidget automatically discards 'move'
        # from the state on button release.
        state_modifier_keys = dict(clear='not-applicable', move_vertex='not-applicable', move_all='not-applicable', move='not-applicable', square='not-applicable', center='not-applicable')
        _SelectorWidget.__init__(self, ax, onselect, useblit=useblit, state_modifier_keys=state_modifier_keys)

        if mark_point is None:
            self._xs = [0]
            self._ys = [0]
        else:
            self._xs = [mark_point[0], mark_point[0]]
            self._ys = [mark_point[1], mark_point[1]]

        self._active_handle_idx = -1
        self._polygon_completed = False
        self.vertex_select_radius = vertex_select_radius

        if lineprops is None:
            lineprops = dict(color='k', linestyle='-', linewidth=2, alpha=0.5)
        lineprops['animated'] = self.useblit
        self.line = Line2D(self._xs, self._ys, **lineprops)
        self.ax.add_line(self.line)

        if markerprops is None:
            markerprops = dict(mec='k', mfc=lineprops.get('color', 'k'))
        self._polygon_handles = ToolHandles(self.ax, self._xs, self._ys, useblit=self.useblit, marker_props=markerprops)
        self.artists = [self.line, self._polygon_handles.artist]
        self.set_visible(True)


    def _press(self, event):
        """Button press event handler"""
        # Check for selection of a tool handle.
        if (self._polygon_completed or 'move_vertex' in self.state) and len(self._xs) > 0:
            h_idx, h_dist = self._polygon_handles.closest(event.x, event.y)
            if h_dist < self.vertex_select_radius:
                self._active_handle_idx = h_idx
        # Save the vertex positions at the time of the press event (needed to
        # support the 'move_all' state modifier).
        self._xs_at_press, self._ys_at_press = self._xs[:], self._ys[:]

    def _release(self, event):
        """Button release event handler"""
        # Release active tool handle.
        if self._active_handle_idx >= 0:
            self._active_handle_idx = -1

        # Complete the polygon.
        elif (len(self._xs) > 3
              and self._xs[-1] == self._xs[0]
              and self._ys[-1] == self._ys[0]):
            self._polygon_completed = True

        # Place new vertex.
        elif (not self._polygon_completed
              and 'move_all' not in self.state
              and 'move_vertex' not in self.state):
            self._xs.insert(-1, event.xdata)
            self._ys.insert(-1, event.ydata)

        if self._polygon_completed:
            self.onselect(self.verts)

    def onmove(self, event):
        """Cursor move event handler and validator"""
        # Method overrides _SelectorWidget.onmove because the polygon selector
        # needs to process the move callback even if there is no button press.
        # _SelectorWidget.onmove include logic to ignore move event if
        # eventpress is None.
        if not self.ignore(event):
            event = self._clean_event(event)
            self._onmove(event)
            return True
        return False

    def _onmove(self, event):
        """Cursor move event handler"""
        # Move the active vertex (ToolHandle).
        if self._active_handle_idx >= 0:
            idx = self._active_handle_idx
            self._xs[idx], self._ys[idx] = event.xdata, event.ydata
            # Also update the end of the polygon line if the first vertex is
            # the active handle and the polygon is completed.
            if idx == 0 and self._polygon_completed:
                self._xs[-1], self._ys[-1] = event.xdata, event.ydata

        # Move all vertices.
        elif 'move_all' in self.state and self.eventpress:
            dx = event.xdata - self.eventpress.xdata
            dy = event.ydata - self.eventpress.ydata
            for k in range(len(self._xs)):
                self._xs[k] = self._xs_at_press[k] + dx
                self._ys[k] = self._ys_at_press[k] + dy

        # Do nothing if completed or waiting for a move.
        elif (self._polygon_completed
              or 'move_vertex' in self.state or 'move_all' in self.state):
            return

        # Position pending vertex.
        else:
            # Calculate distance to the start vertex.
            x0, y0 = self.line.get_transform().transform((self._xs[0],
                                                          self._ys[0]))
            v0_dist = np.sqrt((x0 - event.x) ** 2 + (y0 - event.y) ** 2)
            # Lock on to the start vertex if near it and ready to complete.
            if len(self._xs) > 3 and v0_dist < self.vertex_select_radius:
                self._xs[-1], self._ys[-1] = self._xs[0], self._ys[0]
            else:
                self._xs[-1], self._ys[-1] = event.xdata, event.ydata

        self._draw_polygon()

    def _on_key_press(self, event):
        """Key press event handler"""
        # Remove the pending vertex if entering the 'move_vertex' or
        # 'move_all' mode
        if (not self._polygon_completed
                and ('move_vertex' in self.state or 'move_all' in self.state)):
            self._xs, self._ys = self._xs[:-1], self._ys[:-1]
            self._draw_polygon()

    def _on_key_release(self, event):
        """Key release event handler"""
        # Add back the pending vertex if leaving the 'move_vertex' or
        # 'move_all' mode (by checking the released key)
        if (not self._polygon_completed
                and
                (event.key == self.state_modifier_keys.get('move_vertex')
                 or event.key == self.state_modifier_keys.get('move_all'))):
            self._xs.append(event.xdata)
            self._ys.append(event.ydata)
            self._draw_polygon()
        # Reset the polygon if the released key is the 'clear' key.
        elif event.key == self.state_modifier_keys.get('clear'):
            event = self._clean_event(event)
            self._xs, self._ys = [event.xdata], [event.ydata]
            self._polygon_completed = False
            self.set_visible(True)

    def _draw_polygon(self):
        """Redraw the polygon based on the new vertex positions."""
        self.line.set_data(self._xs, self._ys)
        # Only show one tool handle at the start and end vertex of the polygon
        # if the polygon is completed or the user is locked on to the start
        # vertex.
        if (self._polygon_completed
                or (len(self._xs) > 3
                    and self._xs[-1] == self._xs[0]
                    and self._ys[-1] == self._ys[0])):
            self._polygon_handles.set_data(self._xs[:-1], self._ys[:-1])
        else:
            self._polygon_handles.set_data(self._xs, self._ys)
        self.update()

    @property
    def verts(self):
        """Get the polygon vertices.
        Returns
        -------
        list
            A list of the vertices of the polygon as ``(xdata, ydata)`` tuples.
        """
        return list(zip(self._xs[:-1], self._ys[:-1]))
