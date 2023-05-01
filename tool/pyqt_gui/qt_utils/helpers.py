import os

from PyQt5.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QFrame,
    QTextEdit, QLabel, QScrollArea
)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QIcon

from tool.pyqt_gui.qt_utils.images_grid import ImagesGrid
from tool.core.distance_wrapper import get_neighbours


def get_dir_layout(callback_fn, output_dir_line, label, default_dir, parent):
    text = QLabel(label, parent)
    parent.layout.addWidget(text)

    output_dir_layout = QHBoxLayout()
    select_dir_button = QPushButton()
    select_dir_button.setIcon(QIcon("tool/gui_graphics/select.png"))

    output_dir_line.setEnabled(False)
    output_dir_line.setText(default_dir)
    select_dir_button.clicked.connect(callback_fn)
    output_dir_layout.addWidget(output_dir_line)
    output_dir_layout.addWidget(select_dir_button)
    parent.layout.addLayout(output_dir_layout)


class ShowImageFrame(QFrame):
    def __init__(self, parent):
        super(ShowImageFrame, self).__init__(parent)

        self.setFrameShape(QFrame.StyledPanel)
        self.resize(100, 100)
        self.layout = QHBoxLayout()

        self.left_layout = QVBoxLayout()
        self.img_meta = QTextEdit()
        self.img_meta.setMaximumHeight(150)
        self.img_meta.setReadOnly(True)
        self.left_layout.addWidget(self.img_meta)

        self.image_label = QLabel(self)
        self.left_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addLayout(self.left_layout)

        self.right_layout = QVBoxLayout()
        self.images_grid = ImagesGrid(self)
        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.images_grid)
        self.right_layout.addWidget(self.scroll)
        self.layout.addLayout(self.right_layout)

        self.setLayout(self.layout)

    def show_image(self, info):
        self.img_meta.clear()
        self.img_meta.setText(info.relative_path)
        self.img_meta.append("OoD Score: {:10.3f}".format(info.ood_score))
        strings = ['{:.3f}'.format(x) for x in info.probabilities]
        self.img_meta.append("Probabilities: " + ", ".join(strings))
        self.img_meta.append("Labels: " + ", ".join(info.labels))
        pix = QPixmap(os.path.join(info.absolute_path, info.relative_path))
        if (pix.width() >= pix.height()) and (pix.width() > 800):
            pix = pix.scaledToWidth(800)
        elif (pix.height() > pix.width()) and (pix.height() > 800):
            pix = pix.scaledToHeight(800)
        self.image_label.setPixmap(pix)

        # Show grid with neighbours
        self.images_grid.clear_layout()
        images_meta = get_neighbours.get_k_neighbours(info, k=10)
        self.images_grid.show_images(images_meta, None)


class ImageWindow(QMainWindow):
    def __init__(self, parent=None):
        super(ImageWindow, self).__init__(parent)

        self.image_widget = ShowImageFrame(self)
        self.setCentralWidget(self.image_widget)
        self.setWindowTitle("ImageDisplay")

    def show_image(self, image_info):
        self.image_widget.show_image(image_info)
