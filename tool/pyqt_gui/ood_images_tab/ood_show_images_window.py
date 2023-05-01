from PyQt5.QtWidgets import (
    QMainWindow,
    QDockWidget,
)

from PyQt5.QtCore import Qt

from tool.pyqt_gui.paths_settings_frame import PathsSettingsFrame
from tool.pyqt_gui.analyze_settings import AnalyzeTabSettingsFrame
from tool.pyqt_gui.ood_images_tab.show_images_frame import ShowImagesFrame, HistogramFrame


class AnalyzeTab(QMainWindow):
    def __init__(self, parent):
        super(AnalyzeTab, self).__init__(parent)

        self.path_settings_frame = PathsSettingsFrame(self)
        self.path_settings_frame.setMaximumHeight(150)

        self.common_settings_frame = AnalyzeTabSettingsFrame(self)
        self.common_settings_frame.setMaximumHeight(150)

        self.common_settings_dock = QDockWidget('Settings', self)
        self.common_settings_dock.setWidget(self.common_settings_frame)
        self.common_settings_dock.setFloating(False)

        self.hist_dock = QDockWidget('Histogram', self)
        self.hist_widget = HistogramFrame(self)
        self.hist_dock.setWidget(self.hist_widget)
        self.hist_dock.setFloating(False)

        self.dock = QDockWidget('OoD', self)
        self.widget_1 = ShowImagesFrame(self)
        self.dock.setWidget(self.widget_1)
        self.dock.setFloating(False)

        # Add subscription of all widgets to common setting
        self.common_settings_frame.analyze_settings_changed_signal.connect(self.widget_1.analyze_settings_changed)

        self.path_settings_frame.ood_settings_changed_signal.connect(self.widget_1.settings_changed)
        self.path_settings_frame.ood_settings_changed_signal.connect(self.hist_widget.analyze_settings_changed)

        self.setCentralWidget(self.path_settings_frame)
        self.addDockWidget(Qt.TopDockWidgetArea, self.common_settings_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.hist_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock)

        self.common_settings_frame.emit_settings()
        self.path_settings_frame.emit_settings()
