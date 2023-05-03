import os
import math
import numpy as np

from PyQt5.QtWidgets import (
    QPushButton,
    QGridLayout, QWidget
)

from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import QPixmap, QPainter, QIcon, QColor
from PIL.ImageQt import ImageQt

from tool.pyqt_gui.qt_utils.qt_types import ImageInfo
from typing import Optional, List


class ImagesGrid(QWidget):

    def __init__(self, parent):
        super(ImagesGrid, self).__init__(parent)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

    def clear_layout(self):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    @staticmethod
    def create_label_background(width=50, height=50, color=Qt.white):
        label_background = QPixmap(width, height)
        label_background.fill(color)
        return label_background

    def show_images(self, images_meta: List[ImageInfo],
                    background_colors: Optional[List[QColor]]):

        if len(images_meta) < 1:
            return

        base_size = QSize(110, 110)
        img_size = QSize(100, 100)
        labels = images_meta[0].labels
        images_root_dir = images_meta[0].absolute_path
        if background_colors is None:
            background_colors = [Qt.white for i in range(len(labels))]
        labels_backgrounds = [ImagesGrid.create_label_background(color=background) for background in background_colors]
        transparent_background = ImagesGrid.create_label_background(color=QColor(Qt.transparent))

        columns_count = 3
        for idx in range(len(images_meta)):
            current_column_id = idx % columns_count
            current_row_id = math.floor(idx / columns_count)

            img_full_path = os.path.join(images_root_dir, images_meta[idx].relative_path)
            pixmap = QPixmap(img_full_path).scaled(img_size)

            pixmap_button = QPushButton(self)
            pixmap_button.setFixedSize(base_size)
            pixmap_button.setIconSize(base_size)

            # Add background colors
            label_id = np.argmax(images_meta[idx].probabilities, axis=None)
            if label_id >= len(labels_backgrounds):
                base = transparent_background
            else:
                base = labels_backgrounds[label_id].scaled(base_size)

            painter = QPainter(base)
            painter.drawPixmap(QPoint(5, 5), pixmap)
            pixmap = base
            painter.end()

            pixmap_button.setIcon(QIcon(pixmap))
            pixmap_button.clicked.connect(lambda _, img_meta=images_meta[idx]: self.__show_path(img_meta))
            self.layout.addWidget(pixmap_button, current_row_id, current_column_id)

    def __show_path(self, img_meta):
        from tool.pyqt_gui.qt_utils.helpers import ImageWindow

        show_image_window = ImageWindow(self)
        show_image_window.show_image(img_meta)
        show_image_window.show()

        self.__show_grad(img_meta)

    def __show_grad(self, img_meta):
        from tool.pyqt_gui.qt_utils.helpers import SimpleImageWindow

        grads_folder = os.path.join(img_meta.metadata_dir, "grads")
        if os.path.exists(grads_folder):
            grads_image_path = os.path.join(grads_folder, img_meta.relative_path)
            if os.path.isfile(grads_image_path):
                simple_image_window = SimpleImageWindow("Gradient", self)
                simple_image_window.show_image(grads_image_path)
                simple_image_window.show()
