import sys
import qdarktheme

from PyQt5.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QMainWindow,
    QTabWidget,
    QDockWidget,
)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from tool.metadata_frame import MetadataFrame
from tool.embedder_frame import EmbeddersFrame
from tool.classifiers_frame import OoDFrame
from tool.ood_settings_frame import OoDSettingsFrame
from tool.analyze_settings import AnalyzeTabSettingsFrame
from tool.show_images_frame import ShowImagesFrame, HistogramFrame
from tool.plot_frame import PlotFrame, PlotLegendFrame
from tool.qt_utils.helpers import ShowImageFrame


class AnalyzeTab(QMainWindow):
    def __init__(self, parent):
        super(AnalyzeTab, self).__init__(parent)

        self.common_settings_frame = AnalyzeTabSettingsFrame(self)
        self.common_settings_frame.setMaximumHeight(150)

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
        self.common_settings_frame.analyze_settings_changed_signal.connect(self.hist_widget.analyze_settings_changed)

        self.setCentralWidget(self.common_settings_frame)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.hist_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock)


class PlotTab(QMainWindow):
    def __init__(self, parent):
        super(PlotTab, self).__init__(parent)

        self.common_settings_frame = AnalyzeTabSettingsFrame(self)
        self.common_settings_frame.setMaximumHeight(150)

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
        self.common_settings_frame.analyze_settings_changed_signal.connect(self.plot_widget.inputs_changed)
        self.plot_widget.image_path_signal.connect(self.show_image_widget.show_image)

        self.legend_widget.plot_legend_signal.connect(self.plot_widget.highlight_test_samples_req_changed)

        self.setCentralWidget(self.common_settings_frame)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_legend)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock2)


class CalculationTab(QWidget):
    def __init__(self, parent):
        super(CalculationTab, self).__init__(parent)

        space_between_frames = 10

        self.layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.common_settings_frame = OoDSettingsFrame(self)
        self.widget_1 = MetadataFrame(self)
        self.left_layout.addWidget(self.common_settings_frame)
        self.left_layout.addWidget(self.widget_1)
        self.layout.addLayout(self.left_layout)

        self.widget_2 = EmbeddersFrame(self)
        self.layout.addSpacing(space_between_frames)
        self.layout.addWidget(self.widget_2)

        self.layout.addSpacing(space_between_frames)
        self.classifiers_widget = OoDFrame(self)
        self.layout.addWidget(self.classifiers_widget)

        # Add subscription of all widgets to common setting
        self.common_settings_frame.ood_settings_changed_signal.connect(self.widget_1.ood_settings_changed)
        self.common_settings_frame.ood_settings_changed_signal.connect(self.widget_2.ood_settings_changed)
        self.common_settings_frame.ood_settings_changed_signal.connect(self.classifiers_widget.ood_settings_changed)
        self.widget_1.ood_settings_changed_signal.connect(self.common_settings_frame.ood_settings_changed)

        self.common_settings_frame.emit_settings()

        self.setLayout(self.layout)


class AppWidgets(QWidget):
    def __init__(self, parent):
        super(AppWidgets, self).__init__(parent)

        self.layout = QHBoxLayout()

        self.tabs = QTabWidget()
        self.calculation_tab = CalculationTab(self)
        self.analyze_tab = AnalyzeTab(self)
        self.plot_tab = PlotTab(self)

        self.tabs.addTab(self.calculation_tab, "Calculate")
        self.tabs.addTab(self.analyze_tab, "Analyze")
        self.tabs.addTab(self.plot_tab, "Plot")

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)


class App(QMainWindow):
    def __init__(self):
        """Constructor."""
        super(App, self).__init__()
        self.setWindowTitle("OoD Tool")
        self.setWindowIcon(QIcon('tool/gui_graphics/ood_logo.png'))

        widget = AppWidgets(self)
        widget.setMinimumWidth(1000)
        self.setCentralWidget(widget)
        self.show()


def ood_tool_app():
    app = QApplication(sys.argv)
    window = App()
    app.setStyleSheet(qdarktheme.load_stylesheet())
    window.show()
    sys.exit(app.exec_())

