import os

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QLineEdit,
    QLabel, QMessageBox, QWidget
)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from tool.pyqt_gui.ood_settings import OoDSettings


class ProbabilitiesFromPickleFrame(QWidget):
    def __init__(self, parent, settings: OoDSettings):
        super(ProbabilitiesFromPickleFrame, self).__init__(parent)

        self.probabilities_pkl_file = ''
        self.settings = settings

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.resize(100, 100)
        self.layout = QVBoxLayout()
        self.__probabilities_pkl_layout()

        self.setLayout(self.layout)

    def get_probabilities_pkl_file(self):
        return self.probabilities_pkl_file

    def __probabilities_pkl_layout(self):
        text = QLabel("Select file with embeddings: ", self)
        self.layout.addWidget(text)

        probabilities_pkl_layout = QHBoxLayout()
        select_pkl_file_button = QPushButton()
        select_pkl_file_button.setIcon(QIcon("tool/pyqt_gui/gui_graphics/select.png"))
        self.pkl_probabilities_file_line = QLineEdit()
        self.pkl_probabilities_file_line.setEnabled(False)
        select_pkl_file_button.clicked.connect(self.__get_probabilities_pkl_file)
        probabilities_pkl_layout.addWidget(self.pkl_probabilities_file_line)
        probabilities_pkl_layout.addWidget(select_pkl_file_button)
        self.layout.addLayout(probabilities_pkl_layout)

    def __get_probabilities_pkl_file(self):
        self.probabilities_pkl_file = QFileDialog.getOpenFileName(self, 'Open file',
                                                                  'c:\\', "Probabilities (*.pkl)",
                                                                  options=QFileDialog.DontUseNativeDialog)[0]
        if not os.path.isfile(self.probabilities_pkl_file):
            QMessageBox.warning(self, "", "Pkl file doesn't exist")
        else:
            self.pkl_probabilities_file_line.setText(self.probabilities_pkl_file)

    def ood_settings_changed(self, settings):
        self.settings = settings