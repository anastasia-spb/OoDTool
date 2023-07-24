from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
    QScrollArea
)

from PyQt5.QtCore import Qt

from oodtool.pyqt_gui.qt_utils.images_grid import ImagesGrid

from PyQt5.QtCore import pyqtSignal

from oodtool.pyqt_gui.qt_utils.qt_types import ImageInfo
from oodtool.pyqt_gui.data_loader.loader import DataLoader


class ShowImagesFrame(QWidget):
    _export_signal = pyqtSignal()
    _selected_image_meta_signal = pyqtSignal(ImageInfo)

    def __init__(self, parent, data_loader: DataLoader, with_export=True):
        super(ShowImagesFrame, self).__init__(parent)

        self.legend = None
        self.data_loader = data_loader

        self.layout = QVBoxLayout()
        self.images_widget = ImagesGrid(self)
        self.images_widget._selected_image_meta_signal.connect(self._selected_image_meta_signal.emit)

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.images_widget)
        self.layout.addWidget(self.scroll)

        if with_export:
            export_button = QPushButton("Export")
            export_button.clicked.connect(self.__export_images)
            self.layout.addWidget(export_button)

        self.setLayout(self.layout)

    def __export_images(self):
        self._export_signal.emit()

    def legend_updated(self, legend):
        self.legend = legend

    def show_images(self, images_meta):
        self.images_widget.show_images(images_meta, self.legend)

    def show_neigbours(self, image_meta: ImageInfo):
        images_meta = self.data_loader.get_k_neighbours(image_meta)
        if images_meta is not None:
            self.show_images(images_meta)


