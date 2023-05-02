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
from tool.core.classifier_wrappers.classifier_pipeline import ClassifierPipeline, CLASSIFIER_WRAPPERS
from tool.pyqt_gui.ood_entropy_tab.classifier_widget.classifier_parameters_frame import ClassifierParamsFrame
from tool.pyqt_gui.paths_settings import PathsSettings


class ClassifierFrame(QFrame):
    request_selected_embeddings_files_signal = pyqtSignal()

    def __init__(self, parent):
        super(ClassifierFrame, self).__init__(parent)

        self.settings = PathsSettings()
        self.classifier_output_file = '...'
        self.ood_output = '...'

        default_classifier = list(CLASSIFIER_WRAPPERS.keys())[0]
        self.classifier = ClassifierPipeline(default_classifier)

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
        self.classifier = ClassifierPipeline(value)
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

        self.output_file = QLineEdit(self.classifier_output_file)
        self.output_file.setStyleSheet("color: gray")
        self.output_file.returnPressed.connect(self.__set_embeddings_file_back)
        self.layout.addWidget(self.output_file)

        self.ood_file_line = QLineEdit(self.ood_output)
        self.ood_file_line.setStyleSheet("color: gray")
        self.ood_file_line.returnPressed.connect(self.__set_ood_file_back)
        self.layout.addWidget(self.ood_file_line)

    def __add_params(self):
        self.classifier_params_widget.add_outer_frame(self.classifier.input_hint())

    def __set_embeddings_file_back(self):
        self.output_file.setText(self.classifier_output_file)

    def __set_ood_file_back(self):
        self.ood_file_line.setText(self.ood_output)

    def __calculate_embeddings(self):
        self.calculate_embeddings_button.setEnabled(False)
        self.request_selected_embeddings_files_signal.emit()

    def process_embeddings_files(self, embeddings_files: list):
        self.classifier_output_file = self.classifier.run(embeddings_files=embeddings_files,
                                                          output_dir=self.settings.metadata_folder,
                                                          use_gt_for_training=True, probabilities_file=None,
                                                          kwargs=self.classifier_params_widget.get_parameters())
        self.output_file.setText(self.classifier_output_file)

        if os.path.isfile(self.classifier_output_file):
            pipeline = OoDScore()
            self.ood_output = pipeline.run(self.classifier_output_file, self.settings.metadata_folder)
            self.ood_file_line.setText(self.ood_output)

        self.calculate_embeddings_button.setEnabled(True)