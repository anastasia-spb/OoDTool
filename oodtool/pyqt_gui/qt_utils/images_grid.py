import os
import math
import logging

from PyQt5.QtWidgets import (
    QPushButton,
    QGridLayout, QWidget
)

from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import QPixmap, QPainter, QIcon, QColor

from oodtool.pyqt_gui.qt_utils.qt_types import ImageInfo
from typing import Optional, List, Dict

from PyQt5.QtCore import pyqtSignal

from oodtool.pyqt_gui.ood_images_tab.legend_frame import create_label_background


class ImagesGrid(QWidget):
    _selected_image_meta_signal = pyqtSignal(ImageInfo)

    def __init__(self, parent):
        super(ImagesGrid, self).__init__(parent)

        self.parent = parent

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.object_size = QSize(110, 110)

        self.columns_count = 4
        self.min_columns_count = 4

        self.buttons = []

    def __clear_layout(self):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.buttons = []

    def show_images(self, images_meta: List[ImageInfo],
                    background_colors: Optional[Dict[str, QColor]],
                    columns_count=4):
        self.__clear_layout()

        if len(images_meta) < 1:
            return

        self.columns_count = max(self.min_columns_count, columns_count)

        base_size = self.object_size
        img_size = QSize(100, 100)
        labels = images_meta[0].labels
        if background_colors is None:
            background_colors = {label: Qt.white for label in labels}
        labels_backgrounds = {label: create_label_background(color=background_colors[label]) for label in
                              labels}
        transparent_background = create_label_background(color=QColor(Qt.transparent))

        for idx in range(len(images_meta)):
            img_full_path = os.path.join(images_meta[idx].dataset_root_dir, images_meta[idx].relative_path)
            if not os.path.isfile(img_full_path):
                logging.info("Image path is incorrect %s", img_full_path)
                continue

            current_column_id = idx % columns_count
            current_row_id = math.floor(idx / columns_count)

            pixmap = QPixmap(img_full_path).scaled(img_size)

            pixmap_button = QPushButton(self)
            pixmap_button.setFixedSize(base_size)
            pixmap_button.setIconSize(base_size)

            label = images_meta[idx].gt_label
            if label is not None:
                base = labels_backgrounds[label].scaled(base_size)
            else:
                base = transparent_background.scaled(base_size)

            painter = QPainter(base)
            painter.drawPixmap(QPoint(5, 5), pixmap)
            pixmap = base
            painter.end()

            pixmap_button.setIcon(QIcon(pixmap))
            pixmap_button.clicked.connect(lambda _, img_meta=images_meta[idx]: self.__show_path(img_meta))
            self.buttons.append(pixmap_button)
            self.layout.addWidget(self.buttons[-1], current_row_id, current_column_id)

    def __show_path(self, img_meta):
        self._selected_image_meta_signal.emit(img_meta)

    def __reshape_grid(self, columns_count):
        while self.layout.count():
            _ = self.layout.takeAt(0)

        for idx in range(len(self.buttons)):
            current_column_id = idx % columns_count
            current_row_id = math.floor(idx / columns_count)
            self.layout.addWidget(self.buttons[idx], current_row_id, current_column_id)

    def resizeEvent(self, event):
        if len(self.buttons) > 0:
            current_widget_width = self.size().width()
            columns_count = math.floor(current_widget_width / self.object_size.width()) - 1
            columns_count = max(self.min_columns_count, columns_count)
            if columns_count != self.columns_count:
                self.columns_count = columns_count
                self.__reshape_grid(columns_count)

        return super(ImagesGrid, self).resizeEvent(event)
