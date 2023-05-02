import os

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QHBoxLayout, QLabel, QFrame, QCheckBox, QComboBox
)

from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal

from tool.pyqt_gui.paths_settings import PathsSettings
from tool.pyqt_gui.paths_settings_frame import PathsSettingsFrame
from tool.pyqt_gui.ood_entropy_tab.classifier_widget.classifier_window import ClassifierFrame


class EmbeddingsFilesFrame(QFrame):
    selected_files_signal = pyqtSignal(list)

    def __init__(self, parent):
        super(EmbeddingsFilesFrame, self).__init__(parent)
        self.settings = PathsSettings()
        self.files = []
        self.selected_probabilities = ''
        self.checkboxes = []

        self.setFrameShape(QFrame.StyledPanel)
        # self.setMaximumHeight(200)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

    def ood_settings_changed(self, settings):
        self.settings = settings
        self.__get_all_embeddings_files()
        self.clear_layout()
        for file in self.files:
            select_file_box = QCheckBox(file)
            select_file_box.setChecked(True)
            self.checkboxes.append(select_file_box)
            self.layout.addWidget(self.checkboxes[-1], alignment=Qt.AlignmentFlag.AlignTop)

        text = QLabel("Select file GT for train", self)
        self.layout.addWidget(text, alignment=Qt.AlignmentFlag.AlignTop)

        models_combobox = QComboBox()
        models_combobox.currentTextChanged.connect(self.__on_model_type_change)
        models_combobox.addItems(["Use GT", *self.files])
        self.layout.addWidget(models_combobox, alignment=Qt.AlignmentFlag.AlignTop)

    def __on_model_type_change(self, value):
        self.selected_probabilities = value

    def clear_layout(self):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def get_selected_files(self):
        selected_files = []
        for i, box in enumerate(self.checkboxes):
            if box.isChecked():
                selected_files.append(os.path.join(self.settings.metadata_folder, self.files[i]))
        return selected_files

    def emit_selected_files(self):
        selected_files = self.get_selected_files()
        selected_files.append(self.selected_probabilities)
        self.selected_files_signal.emit(selected_files)

    def __get_all_embeddings_files(self):
        self.files = []
        if not os.path.exists(self.settings.metadata_folder):
            return

        for file in os.listdir(self.settings.metadata_folder):
            if file.endswith(".emb.pkl"):
                self.files.append(file)


class OoDEntropyWindow(QWidget):
    def __init__(self, parent):
        super(OoDEntropyWindow, self).__init__(parent)

        self.layout = QHBoxLayout()

        self.left_layout = QVBoxLayout()

        self.common_settings_frame = PathsSettingsFrame(self)
        self.common_settings_frame.setMaximumHeight(150)
        self.left_layout.addWidget(self.common_settings_frame)

        self.embeddings_file_frame = EmbeddingsFilesFrame(self)
        self.left_layout.addWidget(self.embeddings_file_frame)

        self.layout.addLayout(self.left_layout)

        self.ood_frame = ClassifierFrame(self)
        self.layout.addWidget(self.ood_frame)

        # Add subscription of all widgets to common setting
        self.common_settings_frame.ood_settings_changed_signal.connect(self.embeddings_file_frame.ood_settings_changed)
        self.common_settings_frame.ood_settings_changed_signal.connect(self.ood_frame.ood_settings_changed)

        self.ood_frame.request_selected_embeddings_files_signal.connect(self.embeddings_file_frame.emit_selected_files)
        self.embeddings_file_frame.selected_files_signal.connect(self.ood_frame.process_embeddings_files)

        self.common_settings_frame.emit_settings()

        self.setLayout(self.layout)
