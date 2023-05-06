import os

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QHBoxLayout, QLabel, QFrame, QCheckBox, QComboBox, QLineEdit, QFileDialog, QPushButton
)

from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal

from tool.core.ood_entropy.ood_pipeline import run_ood_pipeline

from tool.pyqt_gui.paths_settings import PathsSettings
from tool.pyqt_gui.paths_settings_frame import PathsSettingsFrame
from tool.pyqt_gui.ood_entropy_tab.classifier_widget.classifier_window import ClassifierFrame
from tool.pyqt_gui.qt_utils import helpers
from tool.pyqt_gui.tools_tab.tools_window import LogoFrame


class EmbeddingsFilesFrame(QFrame):
    selected_files_signal = pyqtSignal(list)

    def __init__(self, parent):
        super(EmbeddingsFilesFrame, self).__init__(parent)
        self.settings = PathsSettings()
        self.files = []
        self.selected_probabilities = ''
        self.checkboxes = []

        self.setFrameShape(QFrame.StyledPanel)
        self.setMaximumHeight(200)

        self.layout = QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.addStretch(1)
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

        text_empty = QLabel("   ", self)
        self.layout.addWidget(text_empty, alignment=Qt.AlignmentFlag.AlignTop)

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


class OoDFromConfig(QFrame):

    def __init__(self, parent):
        super(OoDFromConfig, self).__init__(parent)
        self.settings = PathsSettings()

        self.setFrameShape(QFrame.StyledPanel)
        self.setMaximumHeight(300)
        self.layout = QVBoxLayout()

        self.config_file = ''
        self.config_line = QLineEdit()
        helpers.get_dir_layout(self.__get_config_file, self.config_line,
                               "Select configuration file: ", self.config_file, self)

        self.eval_button = QPushButton("Eval")
        self.eval_button.clicked.connect(self.__eval)
        self.layout.addWidget(self.eval_button, alignment=Qt.AlignmentFlag.AlignTop)

        self.resulting_file = ''
        self.result_file_line = QLineEdit(self.resulting_file)
        self.result_file_line.setStyleSheet("color: gray")
        self.result_file_line.returnPressed.connect(self.__set_file_back)
        self.layout.addWidget(self.result_file_line, alignment=Qt.AlignmentFlag.AlignTop)

        self.setLayout(self.layout)

    def __set_file_back(self):
        self.result_file_line.setText(self.resulting_file)

    def __eval(self):
        if not os.path.isfile(self.config_file):
            return

        self.resulting_file = run_ood_pipeline(self.settings.metadata_folder, self.config_file)
        self.result_file_line.setText(self.resulting_file)

    def __get_config_file(self):
        self.config_file = QFileDialog.getOpenFileName(self, 'Open file',
                                                       'c:\\', "Config (*.config.json)",
                                                       options=QFileDialog.DontUseNativeDialog)[0]
        if not os.path.isfile(self.config_file):
            return

        self.config_line.setText(self.config_file)

    def ood_settings_changed(self, settings):
        self.settings = settings


class OoDEntropyWindow(QWidget):
    def __init__(self, parent):
        super(OoDEntropyWindow, self).__init__(parent)

        self.layout = QVBoxLayout()

        # Layout for train
        self.upper_layout = QHBoxLayout()

        self.left_layout = QVBoxLayout()

        logo_frame = LogoFrame(self)
        self.left_layout.addWidget(logo_frame, alignment=Qt.AlignmentFlag.AlignTop)

        self.common_settings_frame = PathsSettingsFrame(self)
        self.common_settings_frame.setMaximumHeight(200)
        self.left_layout.addWidget(self.common_settings_frame, alignment=Qt.AlignmentFlag.AlignTop)

        self.embeddings_file_frame = EmbeddingsFilesFrame(self)
        self.left_layout.addWidget(self.embeddings_file_frame, alignment=Qt.AlignmentFlag.AlignTop)

        self.upper_layout.addLayout(self.left_layout)

        self.ood_frame = ClassifierFrame(self)
        self.upper_layout.addWidget(self.ood_frame, alignment=Qt.AlignmentFlag.AlignTop)

        self.layout.addLayout(self.upper_layout)

        # self.layout.addSpacing(10)

        # Layout for eval
        empty_frame = QFrame(self)
        empty_frame.setMinimumHeight(10)
        empty_frame.setMaximumHeight(10)

        self.layout.addWidget(empty_frame)

        empty_frame_styled = QFrame(self)
        empty_frame_styled.setFrameShape(QFrame.StyledPanel)
        empty_frame_styled.setMinimumHeight(5)
        empty_frame_styled.setMaximumHeight(5)

        self.layout.addWidget(empty_frame_styled)

        eval_text = QLabel("Evaluation from pretrained config", self)
        eval_text.setStyleSheet("font-weight: bold")
        self.layout.addWidget(eval_text, alignment=Qt.AlignmentFlag.AlignCenter)

        ood_eval_frame = OoDFromConfig(self)
        ood_eval_frame.setMinimumWidth(400)
        ood_eval_frame.setMinimumHeight(250)
        self.layout.addWidget(ood_eval_frame, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add subscription of all widgets to common setting
        self.common_settings_frame.ood_settings_changed_signal.connect(self.embeddings_file_frame.ood_settings_changed)
        self.common_settings_frame.ood_settings_changed_signal.connect(self.ood_frame.ood_settings_changed)

        self.ood_frame.request_selected_embeddings_files_signal.connect(self.embeddings_file_frame.emit_selected_files)
        self.embeddings_file_frame.selected_files_signal.connect(self.ood_frame.process_embeddings_files)

        self.common_settings_frame.emit_settings()

        self.setLayout(self.layout)
