import os

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QFrame,
    QLineEdit,
    QLabel,
    QMessageBox
)

from tool.pyqt_gui.tools_tab.sampler_widget.sampler_thread import SamplerThread
from tool.pyqt_gui.paths_settings import PathsSettings
from tool.pyqt_gui.qt_utils import find_pkl


class SamplerFrame(QFrame):

    def __init__(self, parent):
        super(SamplerFrame, self).__init__(parent)

        spacing_between_layouts = 30
        self.settings = PathsSettings()

        self.output_file = ''
        self.n_samples = 300

        self.setFrameShape(QFrame.StyledPanel)
        self.setMaximumHeight(200)
        self.resize(100, 100)
        self.layout = QVBoxLayout()

        text = QLabel("Density based Sampler", self)
        font = text.font()
        font.setBold(True)
        text.setFont(font)
        self.layout.addWidget(text)

        self.samples_num_line = QLineEdit(str(self.n_samples))
        self.layout.addWidget(self.samples_num_line)
        self.samples_num_line.returnPressed.connect(self.__on_values_change)

        self.layout.addSpacing(spacing_between_layouts)
        self.__add_generate_metadata_button()
        self.layout.addSpacing(spacing_between_layouts)

        self.__add_output_line()

        self.setLayout(self.layout)

    def ood_settings_changed(self, settings):
        self.settings = settings

    def __set_embeddings_file_back(self):
        self.output_file_line.setText(self.output_file)

    def __add_output_line(self):
        self.output_file_line = QLineEdit(self.output_file)
        self.output_file_line.setStyleSheet("color: gray")
        self.output_file_line.returnPressed.connect(self.__set_embeddings_file_back)
        self.layout.addWidget(self.output_file_line)

    def __on_values_change(self):
        t = self.samples_num_line.text()
        try:
            value = int(t)
        except ValueError:
            value = 300
        self.n_samples = value
        self.samples_num_line.setText(str(self.n_samples))

    def __add_generate_metadata_button(self):
        self.generate_button = QPushButton("Sample")
        self.generate_button.setEnabled(True)
        self.generate_button.clicked.connect(self.__sample)
        self.layout.addWidget(self.generate_button)

    def __sample(self):
        self.generate_button.setEnabled(False)
        embeddings_file = find_pkl.get_embeddings_file(self.settings.metadata_folder)
        if not os.path.isfile(embeddings_file):
            QMessageBox.warning(self, "Sampler widget", "Embeddings file not found.")
            self.generate_button.setEnabled(True)
            return

        ood_file = find_pkl.get_ood_file(self.settings.metadata_folder)
        if not os.path.isfile(ood_file):
            QMessageBox.warning(self, "Sampler widget", "OoD file not found.")
            self.generate_button.setEnabled(True)
            return

        self.sampler_thread = SamplerThread(
            embeddings_file=embeddings_file,
            ood_file=ood_file,
            n_samples=self.n_samples)

        self.sampler_thread._signal.connect(self.signal_accept)
        self.sampler_thread.start()

    def signal_accept(self, msg):
        if msg == 0:
            self.output_file_line.setText(self.sampler_thread.result)
        else:
            QMessageBox.warning(self, "Sampler", "Failed to sampler")

        self.generate_button.setEnabled(True)
