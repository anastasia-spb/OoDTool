from typing import Dict, Optional

from PyQt5.QtWidgets import (
    QMainWindow,
    QDockWidget,
)

from PyQt5.QtCore import pyqtSignal, Qt

from oodtool.pyqt_gui.plot_window.plot_settings import AnalyzeTabSettingsFrame

from oodtool.pyqt_gui.plot_window.plot_frame import PlotFrame

from PyQt5.QtGui import QColor
from oodtool.pyqt_gui.qt_utils.qt_types import ImageInfo

from oodtool.pyqt_gui.ood_images_tab.filters_frame import FilterSettings


class PlotWindow(QMainWindow):
    image_path_signal = pyqtSignal(ImageInfo)

    def __init__(self, parent, data_loader, legend: Optional[Dict[str, QColor]] = None,
                 filter_settings: Optional[FilterSettings] = None, head_idx: int = 0):
        super(PlotWindow, self).__init__(parent)
        self.setWindowTitle("Plot")

        self.common_settings_frame = AnalyzeTabSettingsFrame(self)
        self.common_settings_frame.setMaximumHeight(150)

        self.common_settings_dock = QDockWidget('Settings', self)
        self.common_settings_dock.setWidget(self.common_settings_frame)
        self.common_settings_dock.setFloating(False)

        self.plot_widget = PlotFrame(self, data_loader, legend, filter_settings, head_idx)

        # Add subscription of all widgets to common setting
        self.common_settings_frame.analyze_settings_changed_signal.connect(self.plot_widget.analyze_inputs_changed)

        self.setCentralWidget(self.plot_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.common_settings_dock)

        self.common_settings_frame.emit_settings()

    def reload(self, data_loader, legend: Optional[Dict[str, QColor]] = None,
               filter_settings: Optional[FilterSettings] = None):
        self.plot_widget.reload_plot(data_loader, legend, filter_settings)
