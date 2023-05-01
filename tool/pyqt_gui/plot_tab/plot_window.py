from PyQt5.QtWidgets import (
    QMainWindow,
    QDockWidget,
)

from PyQt5.QtCore import Qt

from tool.pyqt_gui.paths_settings_frame import PathsSettingsFrame
from tool.pyqt_gui.analyze_settings import AnalyzeTabSettingsFrame

from tool.pyqt_gui.plot_tab.plot_frame import PlotFrame, PlotLegendFrame
from tool.pyqt_gui.qt_utils.helpers import ShowImageFrame


class PlotTab(QMainWindow):
    def __init__(self, parent):
        super(PlotTab, self).__init__(parent)

        self.path_settings_frame = PathsSettingsFrame(self)
        self.path_settings_frame.setMaximumHeight(150)

        self.common_settings_frame = AnalyzeTabSettingsFrame(self)
        self.common_settings_frame.setMaximumHeight(150)

        self.common_settings_dock = QDockWidget('Settings', self)
        self.common_settings_dock.setWidget(self.common_settings_frame)
        self.common_settings_dock.setFloating(False)

        self.dock = QDockWidget('Embeddings', self)
        self.plot_widget = PlotFrame(self)
        self.dock.setWidget(self.plot_widget)
        self.dock.setFloating(False)

        self.dock_legend = QDockWidget('', self)
        self.legend_widget = PlotLegendFrame(self)
        self.dock_legend.setWidget(self.legend_widget)
        self.dock_legend.setFloating(False)

        self.dock2 = QDockWidget('', self)
        self.show_image_widget = ShowImageFrame(self)
        self.dock2.setWidget(self.show_image_widget)
        self.dock2.setFloating(False)

        # Add subscription of all widgets to common setting
        self.common_settings_frame.analyze_settings_changed_signal.connect(self.plot_widget.analyze_inputs_changed)
        self.path_settings_frame.ood_settings_changed_signal.connect(self.plot_widget.inputs_changed)

        self.plot_widget.image_path_signal.connect(self.show_image_widget.show_image)
        self.legend_widget.plot_legend_signal.connect(self.plot_widget.highlight_test_samples_req_changed)

        self.setCentralWidget(self.path_settings_frame)

        self.addDockWidget(Qt.TopDockWidgetArea, self.common_settings_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_legend)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock2)

        self.common_settings_frame.emit_settings()
        self.path_settings_frame.emit_settings()
