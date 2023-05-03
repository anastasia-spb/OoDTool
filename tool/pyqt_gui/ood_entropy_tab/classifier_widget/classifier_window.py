import os

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLineEdit,
    QComboBox,
    QLabel,
    QMessageBox,
    QScrollArea,
    QTextEdit,
    QCheckBox
)

from PyQt5.QtCore import Qt, pyqtSignal

from tool.core.ood_entropy.ood_score import OoDScore
from tool.core.classifier_wrappers.classifier_pipeline import CLASSIFIER_WRAPPERS
from tool.pyqt_gui.ood_entropy_tab.classifier_widget.classifier_thread import ClassifierThread
from tool.pyqt_gui.ood_entropy_tab.classifier_widget.classifier_parameters_frame import ClassifierParamsFrame
from tool.pyqt_gui.paths_settings import PathsSettings


class ClassifierFrame(QFrame):
    request_selected_embeddings_files_signal = pyqtSignal()

    def __init__(self, parent):
        super(ClassifierFrame, self).__init__(parent)

        self.settings = PathsSettings()
        self.ood_output = '...'

        self.selected_classifier_tag = list(CLASSIFIER_WRAPPERS.keys())[0]
        self.classifier = ClassifierThread(self.selected_classifier_tag,
                                           embeddings_files=[],
                                           output_dir='',
                                           use_gt_for_training=True,
                                           probabilities_file=None,
                                           kwargs=[])

        self.setFrameShape(QFrame.StyledPanel)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.layout = QVBoxLayout()

        self.classifier_params_widget = ClassifierParamsFrame(self.classifier.input_hint(), self)
        self.__select_model_layout()

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.classifier_params_widget)
        self.scroll.setMinimumHeight(600)
        self.layout.addWidget(self.scroll)

        self.__add_evaluate_button()

        self.layout.addStretch()
        self.setLayout(self.layout)

    def ood_settings_changed(self, settings):
        self.settings = settings

    def __select_model_layout(self):
        model_layout = QHBoxLayout()
        models_combobox = QComboBox()
        models_combobox.currentTextChanged.connect(self.__on_model_type_change)
        models_combobox.addItems(CLASSIFIER_WRAPPERS.keys())
        model_layout.addWidget(models_combobox)
        self.layout.insertLayout(0, model_layout)

    def __on_model_type_change(self, value):
        self.selected_classifier_tag = value
        self.classifier = ClassifierThread(self.selected_classifier_tag,
                                           embeddings_files=[],
                                           output_dir='',
                                           use_gt_for_training=True,
                                           probabilities_file=None,
                                           kwargs=[])
        self.classifier_params_widget.reset(self.classifier.input_hint())

    def __add_evaluate_button(self):
        buttons_layout = QHBoxLayout()

        add_params_button = QPushButton("Add")
        add_params_button.clicked.connect(self.__add_params)
        buttons_layout.addWidget(add_params_button)

        self.calculate_embeddings_button = QPushButton("Get Score")
        self.calculate_embeddings_button.clicked.connect(self.__calculate_embeddings)
        buttons_layout.addWidget(self.calculate_embeddings_button)

        self.layout.addLayout(buttons_layout)

        self.ood_file_line = QLineEdit(self.ood_output)
        self.ood_file_line.setStyleSheet("color: gray")
        self.ood_file_line.returnPressed.connect(self.__set_ood_file_back)
        self.layout.addWidget(self.ood_file_line)

    def __add_params(self):
        self.classifier_params_widget.add_outer_frame(self.classifier.input_hint())

    def __set_ood_file_back(self):
        self.ood_file_line.setText(self.ood_output)

    def __calculate_embeddings(self):
        self.calculate_embeddings_button.setEnabled(False)
        self.request_selected_embeddings_files_signal.emit()

    def process_embeddings_files(self, embeddings_files: list):
        if len(embeddings_files) < 2:
            self.calculate_embeddings_button.setEnabled(True)
            return

        self.classifier = ClassifierThread(classifier_tag=self.selected_classifier_tag,
                                           embeddings_files=embeddings_files[:-1],
                                           output_dir=self.settings.metadata_folder,
                                           use_gt_for_training=(embeddings_files[-1] == "Use GT"),
                                           probabilities_file=embeddings_files[-1],
                                           kwargs=self.classifier_params_widget.get_parameters())

        self.classifier._signal.connect(self.signal_accept)
        self.classifier.start()

    def signal_accept(self, output_files):
        if len(output_files) > 0:
            pipeline = OoDScore()
            self.ood_output = pipeline.run(output_files, self.settings.metadata_folder)
            self.ood_file_line.setText(self.ood_output)
        else:
            QMessageBox.warning(self, "OoDScore", "Failed")

        self.calculate_embeddings_button.setEnabled(True)
