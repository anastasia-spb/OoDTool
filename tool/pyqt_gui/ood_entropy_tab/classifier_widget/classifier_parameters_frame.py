import os

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QWidget, QTextEdit, QMessageBox
)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt


class ClassifierParamsFrame(QWidget):

    def __init__(self, input_hint, parent):
        super(ClassifierParamsFrame, self).__init__(parent)

        self.uidentifier = 0

        self.layout = QVBoxLayout()
        self.classifiers_instances = dict()
        self.add_outer_frame(input_hint)
        self.layout.addStretch()
        self.setLayout(self.layout)

    def add_outer_frame(self, input_hint):
        self.uidentifier += 1
        unique_id = self.uidentifier + 1

        outer_frame = QFrame()
        outer_frame.setFrameShape(QFrame.StyledPanel)

        outer_frame_layout = QVBoxLayout()
        close_button_layout = QHBoxLayout()
        close_button = QPushButton()
        close_button.setIcon(QIcon("tool/pyqt_gui/gui_graphics/close.png"))
        close_button.clicked.connect(lambda: self.__delete_frame(unique_id))
        close_button_layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        outer_frame_layout.addLayout(close_button_layout)

        kwargs_line = QTextEdit()
        kwargs_line.setMaximumHeight(150)
        kwargs_line.setText(input_hint)

        outer_frame_layout.addWidget(kwargs_line)
        outer_frame.setLayout(outer_frame_layout)

        self.classifiers_instances[unique_id] = outer_frame
        outer_frame.setMaximumHeight(300)
        idx = len(self.classifiers_instances) - 1
        self.layout.insertWidget(idx, self.classifiers_instances[unique_id], alignment=Qt.AlignmentFlag.AlignTop)

    def reset(self, input_hint):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.classifiers_instances = dict()
        self.add_outer_frame(input_hint)

    def __delete_frame(self, idx: int):
        self.classifiers_instances[idx].deleteLater()
        del self.classifiers_instances[idx]

    def get_parameters(self):
        parameters = []
        if not self.classifiers_instances:
            return
        for _, classifier in self.classifiers_instances.items():
            kwargs_line = classifier.findChild(QTextEdit)
            if kwargs_line is not None:
                weight_decay = 0.0
                try:
                    weight_decay = float(kwargs_line.toPlainText())
                except SyntaxError:
                    QMessageBox.critical(self, "", "Parameters couldn't be parsed")
                parameters.append(weight_decay)
        return parameters
