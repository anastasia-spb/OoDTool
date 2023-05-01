import os

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLineEdit,
    QComboBox, QScrollArea, QWidget
)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from tool.ood_settings import OoDSettings
from tool.core.ood_entropy.ood_score import OoDScore
from tool.embedder_frame_widgets import classifier_widget
from tool.classifiers.classifier import CLASSIFIER_WRAPPERS
from tool.qt_utils import find_pkl


class ClassifiersWidget(QWidget):

    def __init__(self, parent):
        super(ClassifiersWidget, self).__init__(parent)

        self.settings = OoDSettings()
        self.uidentifier = 0

        self.layout = QVBoxLayout()
        self.classifiers_instances = dict()
        self.add_outer_frame()
        self.layout.addStretch()
        self.setLayout(self.layout)

    def ood_settings_changed(self, settings):
        self.settings = settings
        if not self.classifiers_instances:
            return
        for _, classifier in self.classifiers_instances.items():
            classifier_instance = classifier.findChild(classifier_widget.ClassifierFrame)
            if classifier_instance is not None:
                classifier_instance.ood_settings_changed(settings)

    def add_outer_frame(self, selected_classifier=list(CLASSIFIER_WRAPPERS.keys())[0]):
        self.uidentifier += 1
        unique_id = self.uidentifier + 1

        outer_frame = QFrame()
        outer_frame.setFrameShape(QFrame.StyledPanel)

        outer_frame_layout = QVBoxLayout()
        close_button_layout = QHBoxLayout()
        close_button = QPushButton()
        close_button.setIcon(QIcon("tool/gui_graphics/close.png"))
        close_button.clicked.connect(lambda: self.__delete_frame(unique_id))
        close_button_layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        outer_frame_layout.addLayout(close_button_layout)
        classifier_frame = classifier_widget.ClassifierFrame(self, selected_classifier, self.settings)
        outer_frame_layout.addWidget(classifier_frame)
        outer_frame.setLayout(outer_frame_layout)

        self.classifiers_instances[unique_id] = outer_frame
        outer_frame.setMaximumHeight(300)
        idx = len(self.classifiers_instances) - 1
        self.layout.insertWidget(idx, self.classifiers_instances[unique_id], alignment=Qt.AlignmentFlag.AlignTop)

    def __delete_frame(self, idx: int):
        self.classifiers_instances[idx].deleteLater()
        del self.classifiers_instances[idx]

    def __run_classifiers(self, embeddings_file):
        all_files = []
        if not self.classifiers_instances:
            return
        for _, classifier in self.classifiers_instances.items():
            classifier_instance = classifier.findChild(classifier_widget.ClassifierFrame)
            if classifier_instance is not None:
                classifier_instance.calculate_probabilities(embeddings_file)
                all_files.extend(classifier_instance.get_probabilities_pkl_file())

        return all_files

    def clear_old_results(self):
        if not self.classifiers_instances:
            return
        for _, classifier in self.classifiers_instances.items():
            classifier_instance = classifier.findChild(classifier_widget.ClassifierFrame)
            if classifier_instance is not None:
                classifier_instance.clear_old_results()

    def run_classification(self, embeddings_file):
        self.clear_old_results()
        return self.__run_classifiers(embeddings_file)


class OoDFrame(QFrame):

    def __init__(self, parent):
        super(OoDFrame, self).__init__(parent)

        self.ood_pkl_file = 'Not calculated...'

        self.settings = OoDSettings()

        self.setFrameShape(QFrame.NoFrame)
        self.resize(100, 100)
        self.layout = QVBoxLayout()
        self.classifiers_widget = ClassifiersWidget(self)
        self.__add_classifiers_menu()

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.classifiers_widget)

        self.layout.addWidget(self.scroll)

        self.__add_ood_button()
        self.__add_ood_output_file()

        self.setLayout(self.layout)

    def __add_classifiers_menu(self):
        menu_combobox = QComboBox()
        menu_combobox.textActivated.connect(self.__on_menu_change)
        menu_combobox.addItems(CLASSIFIER_WRAPPERS.keys())
        self.layout.addWidget(menu_combobox)

    def __on_menu_change(self, value):
        self.classifiers_widget.add_outer_frame(value)

    def __add_ood_button(self):
        self.ood_button = QPushButton("Get OoD Score")
        self.ood_button.clicked.connect(self.__calculate_ood_score)
        self.layout.addWidget(self.ood_button)

    def __calculate_ood_score(self):
        self.ood_button.setEnabled(False)
        self.run(find_pkl.get_embeddings_file(self.settings.metadata_folder))

    def __add_ood_output_file(self):
        self.output_file = QLineEdit(self.ood_pkl_file)
        self.output_file.setStyleSheet("color: gray")
        self.output_file.returnPressed.connect(self.__set_ood_file_back)
        self.layout.addWidget(self.output_file)

    def __set_ood_file_back(self):
        self.output_file.setText(self.ood_pkl_file)

    def ood_settings_changed(self, settings):
        self.settings = settings
        self.classifiers_widget.ood_settings_changed(settings)

    def run(self, embeddings_file):
        probabilities = self.classifiers_widget.run_classification(embeddings_file)
        ood_score = OoDScore(probabilities, self.settings.metadata_folder)
        self.ood_pkl_file = ood_score.run()
        if os.path.isfile(self.ood_pkl_file):
            self.output_file.setText(self.ood_pkl_file)
            self.ood_button.setEnabled(True)
