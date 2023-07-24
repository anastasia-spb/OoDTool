import distinctipy

from typing import List


from PyQt5.QtWidgets import (
    QVBoxLayout,
    QFrame,
    QScrollArea, QLabel, QGridLayout, QWidget, QSpacerItem, QSizePolicy
)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtCore import pyqtSignal


def create_label_background(width=50, height=50, color=Qt.white):
    label_background = QPixmap(width, height)
    label_background.fill(color)
    return label_background


class LegendGrid(QFrame):
    def __init__(self, parent):
        super(LegendGrid, self).__init__(parent)

        self.legend = dict()
        self.parent = parent

        self.layout = QGridLayout()
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

    def __add_label(self, label_color, label_text, row, column):
        label = QLabel(self)
        pixmap = create_label_background(width=25, height=25, color=label_color)
        label.setPixmap(pixmap.scaledToHeight(50))
        label.resize(pixmap.width(),
                     pixmap.height())
        self.layout.addWidget(label, row, column)
        text = QLabel(label_text, self)
        text.setStyleSheet("font: 10pt;")
        # text.setStyleSheet("background-image: url(./{0});".format(label_img))
        self.layout.addWidget(text, row + 1, column)

    def __clean(self):
        self.legend.clear()
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def create_legend(self, labels):
        colorlist = distinctipy.get_colors(len(labels))
        colors_list = [QColor(int(r*255), int(255*g), int(255*b)) for r, g, b in colorlist]
        self.__clean()
        current_row_id = 0
        for idx in range(len(labels)):
            self.__add_label(colors_list[idx], labels[idx], current_row_id, 0)
            current_row_id += 2
            self.legend[labels[idx]] = colors_list[idx]

        vspacer = QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout.addItem(vspacer, 2*len(labels), 0)

        self.parent._legend_signal.emit(self.legend)

    def get_legend(self) -> dict:
        return self.legend


class LegendWidget(QWidget):
    _legend_signal = pyqtSignal(dict)

    def __init__(self, parent):
        super(LegendWidget, self).__init__(parent)

        self.layout = QVBoxLayout()
        self.legend_frame = LegendGrid(self)

        self.scroll = None

        self.setLayout(self.layout)

    def create_legend(self, labels: List[str]):
        if self.scroll is not None:
            self.layout.removeWidget(self.scroll)
            self.scroll.setParent(None)
            self.scroll.deleteLater()
            self.scroll = None

        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.legend_frame)
        self.scroll = scroll
        self.layout.addWidget(self.scroll)

        self.legend_frame.create_legend(labels)

