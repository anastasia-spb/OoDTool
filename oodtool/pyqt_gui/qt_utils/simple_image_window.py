import numpy as np

from PyQt5.QtWidgets import (
    QHBoxLayout,
    QMainWindow, QLabel, QApplication, QWidget
)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage


class ShowSimpleImageFrame(QWidget):
    def __init__(self, parent):
        super(ShowSimpleImageFrame, self).__init__(parent)

        self.pix = None

        self.layout = QHBoxLayout()
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(self.layout)

    def show_image(self, cv_image: np.ndarray):
        height, width, channel = cv_image.shape
        bytesPerLine = 3 * width
        qImg = QImage(cv_image.data, width, height, bytesPerLine, QImage.Format_BGR888)
        pix = QPixmap(qImg)
        self.pix = pix.scaledToWidth(400)
        self.image_label.setPixmap(self.pix)

    def __scale_image(self, up: bool):
        if self.pix is None:
            return

        h = self.pix.height()

        if up:
            max_h = 600
            h = min(max_h, h + 20)
        else:
            min_h = 40
            h = max(min_h, h - 20)

        self.pix = self.pix.scaledToHeight(h)
        self.image_label.setPixmap(self.pix)

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        shift_pressed = (modifiers == Qt.ShiftModifier)

        if shift_pressed:
            if event.angleDelta().y() > 0:
                self.__scale_image(True)
            elif event.angleDelta().y() < 0:
                self.__scale_image(False)


class SimpleImageWindow(QMainWindow):
    def __init__(self, title, subtitle, parent=None):
        super(SimpleImageWindow, self).__init__(parent)

        self.img_frame = ShowSimpleImageFrame(self)
        self.setCentralWidget(self.img_frame)
        self.setWindowTitle(": ".join((title, subtitle)))

    def show_image(self, cv_image: np.ndarray):
        self.img_frame.show_image(cv_image)
