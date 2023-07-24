from dataclasses import dataclass

from typing import List, Optional

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QFrame,
    QCheckBox,
    QRadioButton,
    QHBoxLayout,
    QGridLayout,
    QWidget,
    QLabel,
    QLineEdit,
    QComboBox,
    QMainWindow
)

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt


@dataclass
class FilterSettings:
    selected_labels: List[str]
    show_train: bool
    show_test: bool
    sort_ascending: bool
    num_images_to_show: int
    ood_method_name: Optional[str]

    def __init__(self):
        self.selected_labels = []
        self.show_train = True
        self.show_test = True
        self.sort_ascending = True
        self.num_images_to_show = 300
        self.ood_method_name = None


class LabelsCheckboxesWidget(QWidget):
    _labels_signal = pyqtSignal(list)

    def __init__(self, parent, labels: List[str], selected_labels: List[str]):
        super(LabelsCheckboxesWidget, self).__init__(parent)

        self.layout = QVBoxLayout()
        self.labels = labels

        grid_layout = QGridLayout()
        self.checkboxes = []

        for i, label in enumerate(labels):
            self.checkboxes.append(QCheckBox(label))
            self.checkboxes[-1].stateChanged.connect(self.__get_selected_labels)
            if label in selected_labels:
                self.checkboxes[-1].setChecked(True)
            grid_layout.addWidget(self.checkboxes[-1], i, 0)
        self.layout.addLayout(grid_layout)

        select_layout = QHBoxLayout()

        select_all_button = QPushButton("Select all")
        select_all_button.clicked.connect(self.__select_all)
        select_layout.addWidget(select_all_button)

        select_none_button = QPushButton("Unselect all")
        select_none_button.clicked.connect(self.__select_none)
        select_layout.addWidget(select_none_button)

        self.layout.addLayout(select_layout)
        self.setLayout(self.layout)

    def reset(self, labels: List[str]):
        self.labels = labels
        self.__select_all()

    def __select_none(self):
        for idx, box in enumerate(self.checkboxes):
            box.setChecked(False)
        selected_labels = []
        self._labels_signal.emit(selected_labels)

    def __select_all(self):
        for idx, box in enumerate(self.checkboxes):
            box.setChecked(True)
        selected_labels = self.labels
        self._labels_signal.emit(selected_labels)

    def __get_selected_labels(self):
        selected_labels = []
        for idx, box in enumerate(self.checkboxes):
            if box.isChecked():
                selected_labels.append(self.labels[idx])
        self._labels_signal.emit(selected_labels)


class LabelsCheckboxesWindow(QMainWindow):
    _signal = pyqtSignal(list)

    def __init__(self, parent, labels: List[str], selected_labels: List[str]):
        super(LabelsCheckboxesWindow, self).__init__(parent)

        self.checkboxes_widget = LabelsCheckboxesWidget(self, labels, selected_labels)
        self.setCentralWidget(self.checkboxes_widget)
        self.setWindowTitle("Select labels")

        self.checkboxes_widget._labels_signal.connect(self.reemit)

    def reemit(self, selected_labels):
        self._signal.emit(selected_labels)

    def reset(self, labels: List[str]):
        self.checkboxes_widget.reset(labels)


class FiltersFrame(QFrame):
    def __init__(self, parent):
        super(FiltersFrame, self).__init__(parent)

        self.labels = []
        self.settings = FilterSettings()

        self.layout = QVBoxLayout()

        # Select number of images to show
        self.layout.addSpacing(5)
        num_images_layout = QHBoxLayout()

        num_images_text = QLabel("Max number of images to show/export: ", self)
        num_images_layout.addWidget(num_images_text, alignment=Qt.AlignmentFlag.AlignLeft)

        self.num_images_to_show_line = QLineEdit()
        self.num_images_to_show_line.setText(str(self.settings.num_images_to_show))
        num_images_layout.addWidget(self.num_images_to_show_line, alignment=Qt.AlignmentFlag.AlignLeft)

        self.layout.addLayout(num_images_layout)

        # Filter by Label
        self.layout.addSpacing(15)
        self.labels_window = None
        select_labels_button = QPushButton("Select labels")
        select_labels_button.setMinimumWidth(200)
        select_labels_button.clicked.connect(self.__select_labels)
        self.layout.addWidget(select_labels_button, alignment=Qt.AlignmentFlag.AlignLeft)

        # Filter by Train/Test flag
        self.layout.addSpacing(15)
        train_test_layout = QHBoxLayout()
        self.train_checkbox = QCheckBox("Train")
        self.train_checkbox.setChecked(self.settings.show_train)

        train_test_layout.addWidget(self.train_checkbox)
        self.test_checkbox = QCheckBox("Test")
        self.test_checkbox.setChecked(self.settings.show_test)
        train_test_layout.addWidget(self.test_checkbox)

        self.layout.addLayout(train_test_layout)

        # Filter by OoD Score: ascending, descending
        self.layout.addSpacing(15)
        select_ood_file_layout = QHBoxLayout()
        text = QLabel("Sort by OoD score", self)
        select_ood_file_layout.addWidget(text)
        self.ood_files_combobox = QComboBox()
        select_ood_file_layout.addWidget(self.ood_files_combobox)
        self.layout.addLayout(select_ood_file_layout)

        ood_buttons_layout = QHBoxLayout()
        self.ood_button_a = QRadioButton("ascending")
        self.ood_button_a.setChecked(self.settings.sort_ascending)
        self.ood_button_a.toggled.connect(lambda: self.__ood_checkbox_state_changed())
        ood_buttons_layout.addWidget(self.ood_button_a)

        self.ood_button_d = QRadioButton("descending")
        self.ood_button_d.setChecked(not self.settings.sort_ascending)
        self.ood_button_d.toggled.connect(lambda: self.__ood_checkbox_state_changed())
        ood_buttons_layout.addWidget(self.ood_button_d)

        self.layout.addLayout(ood_buttons_layout)

        self.setLayout(self.layout)

    def __ood_checkbox_state_changed(self):
        if self.ood_button_a.isChecked():
            self.settings.sort_ascending = True
            self.ood_button_d.setChecked(False)
        else:
            self.settings.sort_ascending = False
            self.ood_button_a.setChecked(False)

    def __select_labels(self):
        if (self.labels_window is None) or (not self.labels_window.isVisible()):
            self.labels_window = LabelsCheckboxesWindow(self, self.labels, self.settings.selected_labels)
            self.labels_window._signal.connect(self.save_selected_labels)
        self.labels_window.show()
        self.labels_window.activateWindow()
        self.labels_window.raise_()

    def save_selected_labels(self, selected_labels):
        self.settings.selected_labels = selected_labels

    def save_labels(self, labels: List[str]):
        self.labels = labels
        self.settings.selected_labels = self.labels

        if self.labels_window is not None:
            self.labels_window.reset(self.labels)

    def update_ood_methods(self, available_ood_files: Optional[List[str]] = None):
        self.ood_files_combobox.clear()
        if available_ood_files is not None:
            self.ood_files_combobox.addItems(available_ood_files)

    def get_settings(self):
        t = self.num_images_to_show_line.text()
        try:
            num_images_to_show = min(max(0, int(t)), 10000)
        except ValueError:
            num_images_to_show = self.settings.num_images_to_show

        self.settings.num_images_to_show = num_images_to_show
        self.num_images_to_show_line.setText(str(self.settings.num_images_to_show))

        self.settings.show_test = self.test_checkbox.isChecked()
        self.settings.show_train = self.train_checkbox.isChecked()

        self.settings.ood_method_name = self.ood_files_combobox.currentText()

        return self.settings
