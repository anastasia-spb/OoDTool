import qdarktheme
import logging

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QDockWidget,
    QTabWidget,
    QAction,
    QMenu,
)

from PyQt5.QtGui import QIcon

from PyQt5.QtCore import Qt, QSettings, pyqtSlot, QElapsedTimer

from oodtool.pyqt_gui.ood_images_tab.show_images_frame import ShowImagesFrame
from oodtool.pyqt_gui.ood_images_tab.settings_widget import SettingsWidget
from oodtool.pyqt_gui.ood_images_tab.legend_frame import LegendWidget

from oodtool.pyqt_gui.data_loader.loader import DataLoader


class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.setWindowTitle("OoD Tool")
        self.setWindowIcon(QIcon('oodtool/pyqt_gui/gui_graphics/ood_logo.png'))

        self.data_loader = DataLoader()

        self.settings_dock = QDockWidget("Settings", self)
        self.settings_dock.setObjectName("SettingsDock")
        self.settings_widget = SettingsWidget(self, self.data_loader)
        self.settings_dock.setWidget(self.settings_widget)
        self.settings_dock.setFloating(False)

        self.images_dock = QDockWidget("ImageView", self)
        self.images_dock.setObjectName("ImageViewDock")
        self.widget_1 = ShowImagesFrame(self, self.data_loader)
        self.images_dock.setWidget(self.widget_1)
        self.images_dock.setFloating(False)

        self.neighbours_dock = QDockWidget("NeighboursView", self)
        self.neighbours_dock.setObjectName("NeighboursViewDock")
        self.neighbours_widget = ShowImagesFrame(self, self.data_loader, with_export=False)
        self.neighbours_dock.setWidget(self.neighbours_widget)
        self.neighbours_dock.setFloating(False)

        self.legend_dock = QDockWidget("Legend", self)
        self.legend_dock.setObjectName("LegendDock")
        self.legend_widget = LegendWidget(self)
        self.legend_dock.setWidget(self.legend_widget)
        self.legend_dock.setFloating(False)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.settings_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.images_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.legend_dock)
        self.tabifyDockWidget(self.images_dock, self.neighbours_dock)

        self.setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.TabPosition.North)

        self.splitDockWidget(self.settings_dock, self.images_dock, Qt.Horizontal)
        self.splitDockWidget(self.images_dock, self.legend_dock, Qt.Horizontal)
        # self.resizeDocks([self.settings_dock, self.images_dock, self.legend_dock], [508, 1123, 259], Qt.Horizontal)
        self.images_dock.raise_()

        # Connect docks for data exchange
        self.legend_widget._legend_signal.connect(self.widget_1.legend_updated)
        self.legend_widget._legend_signal.connect(self.settings_widget.legend_updated)

        self.settings_widget._images_signal.connect(self.widget_1.show_images)

        self.settings_widget._labels_signal.connect(self.legend_widget.create_legend)
        self.settings_widget._show_neighbours_signal.connect(self.neighbours_widget.show_neigbours)

        self.widget_1._selected_image_meta_signal.connect(self.settings_widget.show_img_meta)
        self.widget_1._selected_image_meta_signal.connect(self.neighbours_widget.show_neigbours)

        self.neighbours_widget._selected_image_meta_signal.connect(self.settings_widget.show_img_meta)

        self.widget_1._export_signal.connect(self.settings_widget.export_images)

        # Create menu
        self._create_actions()
        self._create_menu_bar()

        self.read_settings()
        self.show()

    def _create_actions(self):
        self.open_ood_window_action = QAction(self)
        self.open_ood_window_action.setText("&OoD Score")
        self.open_ood_window_action.triggered.connect(self.open_ood_window)

        self.find_neighbours_window_action = QAction(self)
        self.find_neighbours_window_action.setText('&Find neighbours')
        self.find_neighbours_window_action.triggered.connect(self.open_neighbours_window)

        self.project_window_action = QAction(self)
        self.project_window_action.setText('&Project Data')
        self.project_window_action.triggered.connect(self.open_projector_window)

        self.plot_window_action = QAction(self)
        self.plot_window_action.setText('&Plot')
        self.plot_window_action.triggered.connect(self.open_plot_window)

    def _create_menu_bar(self):
        menuBar = self.menuBar()

        fileMenu = QMenu("&Tools", self)
        menuBar.addMenu(fileMenu)
        fileMenu.addAction(self.open_ood_window_action)
        fileMenu.addAction(self.find_neighbours_window_action)
        fileMenu.addAction(self.project_window_action)
        fileMenu.addAction(self.plot_window_action)

    def closeEvent(self, event):
        self.settings = QSettings("OoDTool", "App")
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())

        super(App, self).closeEvent(event)

    def read_settings(self):
        self.settings = QSettings("OoDTool", "App")
        if self.settings.value("geometry") is not None:
            self.restoreGeometry(self.settings.value("geometry"))
        if self.settings.value("windowState") is not None:
            self.restoreState(self.settings.value("windowState"))

    @pyqtSlot()
    def open_ood_window(self):
        self.settings_widget.open_ood_window()

    @pyqtSlot()
    def open_neighbours_window(self):
        self.settings_widget.open_neighbours_window()

    @pyqtSlot()
    def open_projector_window(self):
        self.settings_widget.open_projector_window()

    @pyqtSlot()
    def open_plot_window(self):
        self.settings_widget.open_plot_window()


class OoDToolApp(QApplication):
    t = QElapsedTimer()

    def __init__(self, args):
        super(OoDToolApp, self).__init__(args)
        logging.basicConfig(filename='ood_tool.log', level=logging.INFO)
        self.mainWindow = App()
        self.setStyleSheet(qdarktheme.load_stylesheet())
        self.exec_()  # enter event loop

    def notify(self, receiver, event):
        self.t.start()
        ret = QApplication.notify(self, receiver, event)
        if self.t.elapsed() > 1:
            logging.info(f"processing event type {event.type()} for object {receiver.objectName()} "
                         f"took {self.t.elapsed()}ms")
        return ret

