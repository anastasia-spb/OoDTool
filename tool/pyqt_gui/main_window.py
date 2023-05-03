import sys
import qdarktheme

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QMainWindow,
    QTabWidget,
)

from PyQt5.QtGui import QIcon

from tool.pyqt_gui.tools_tab.tools_window import ToolsWindow
from tool.pyqt_gui.ood_entropy_tab.ood_entropy_window import OoDEntropyWindow
from tool.pyqt_gui.ood_images_tab.ood_show_images_window import AnalyzeTab
from tool.pyqt_gui.plot_tab.plot_window import PlotTab


class AppWidgets(QWidget):
    def __init__(self, parent):
        super(AppWidgets, self).__init__(parent)

        self.layout = QHBoxLayout()

        self.tabs = QTabWidget()
        self.tools_tab = ToolsWindow(self)
        self.ood_entropy_tab = OoDEntropyWindow(self)
        self.analyze_tab = AnalyzeTab(self)
        self.plot_tab = PlotTab(self)

        self.tabs.addTab(self.tools_tab, "Tools")
        self.tabs.addTab(self.ood_entropy_tab, "OoD Tune")
        self.tabs.addTab(self.analyze_tab, "OoD Images")
        self.tabs.addTab(self.plot_tab, "Plot")

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)


class App(QMainWindow):
    def __init__(self):
        """Constructor."""
        super(App, self).__init__()
        self.setWindowTitle("OoD Tool")
        self.setWindowIcon(QIcon('tool/pyqt_gui/gui_graphics/ood_logo.png'))

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

