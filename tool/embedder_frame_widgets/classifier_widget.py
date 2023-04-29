import os
from datetime import datetime
import ast

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QLineEdit,
    QCheckBox, QLabel, QMessageBox, QTextEdit, QWidget
)

from PyQt5.QtCore import Qt

from tool.classifiers.classifier import Classifier
from tool.ood_settings import OoDSettings

from tool.qt_utils.helpers import string_from_kwargs


class ClassifierFrame(QWidget):
    def __init__(self, parent, selected_classifier: str, settings: OoDSettings):
        super(ClassifierFrame, self).__init__(parent)

        self.settings = settings
        self.probabilities_pkl_file = []
        self.classifier = Classifier(selected_classifier)
        self.use_gt_for_training = False

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.resize(100, 100)
        self.layout = QVBoxLayout()

        text = QLabel(self.classifier.get_tag(), self)
        self.layout.addWidget(text)

        self.__output_file()

        self.kwargs_line = QLineEdit(self.classifier.input_hint())
        self.kwargs_line.setMinimumHeight(200)
        self.layout.addWidget(self.kwargs_line)

        self.use_gt = QCheckBox("Use GT for training")
        self.use_gt.setChecked(self.use_gt_for_training)
        self.use_gt.stateChanged.connect(lambda: self.__check_box_state_changed())
        self.layout.addWidget(self.use_gt)

        self.setLayout(self.layout)

    def __check_box_state_changed(self):
        self.use_gt_for_training = self.use_gt.isChecked()

    def add_layout(self, layout):
        self.layout.addLayout(layout)

    def insert_widget(self, idx, widget):
        self.layout.insertWidget(idx, widget)

    def get_probabilities_pkl_file(self):
        return self.probabilities_pkl_file

    def ood_settings_changed(self, settings):
        self.settings = settings

    def __output_file(self):
        self.output_file = QTextEdit()
        self.output_file.setMaximumHeight(60)
        self.output_file.setReadOnly(True)
        self.layout.addWidget(self.output_file)

    def calculate_probabilities(self, embeddings_file, store_in_folder=True):
        if self.classifier is not None:
            try:
                kwargs = ast.literal_eval(self.kwargs_line.text())
            except SyntaxError:
                kwargs = dict()
            if not self.classifier.check_input_kwargs(kwargs):
                QMessageBox.critical(self, "Wrong parameters", self.classifier.parameters_hint())
                return
            probabilities_df = self.classifier.run(embeddings_file, self.settings.metadata_folder,
                                                   self.use_gt_for_training, kwargs)
            self.__store_probabilities(probabilities_df, embeddings_file, kwargs, store_in_folder)

    def clear_old_results(self):
        self.probabilities_pkl_file.clear()
        self.output_file.clear()

    def __store_probabilities(self, probabilities_df, embeddings_file: str, kwargs: dict, store_in_folder):
        base = os.path.splitext(os.path.basename(embeddings_file))[0]
        timestamp_str = datetime.now().strftime("%y%m%d_%H%M%S")
        name = "".join((base, timestamp_str, '.clf.pkl'))
        if store_in_folder:
            output_pkl_dir = os.path.join(self.settings.metadata_folder,
                                          string_from_kwargs(self.classifier.get_tag(), kwargs))
            if not os.path.exists(output_pkl_dir):
                os.makedirs(output_pkl_dir)
            file = os.path.join(output_pkl_dir, name)
        else:
            name = "".join((string_from_kwargs(self.classifier.get_tag(), kwargs), name))
            file = os.path.join(self.settings.metadata_folder, name)

        probabilities_df.to_pickle(file)
        self.output_file.append(file)
        self.probabilities_pkl_file.append(file)

