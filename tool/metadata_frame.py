import os
import pandas as pd
import random

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QFrame,
    QLineEdit,
    QComboBox, QLabel, QMessageBox, QTableView,
)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QAbstractTableModel, pyqtSignal

from tool.metadata_utils.parse_datasets_description import get_datasets_names
from tool.metadata_utils.generate_datasets_metadata import generate_metadata
from tool.ood_settings import OoDSettings
from tool.data_types import types


class TableModel(QAbstractTableModel):
    def __init__(self, data, columns_names):
        super(TableModel, self).__init__()
        self._data = data
        self.columns_names = columns_names

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.columns_names[section]
        return super().headerData(section, orientation, role)

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        return len(self._data)

    def columnCount(self, index):
        return len(self._data[0])


class MetadataFrame(QFrame):
    ood_settings_changed_signal = pyqtSignal(OoDSettings)

    def __init__(self, parent):
        super(MetadataFrame, self).__init__(parent)

        self.settings = OoDSettings()
        self.metadata_file = None

        spacing_between_layouts = 30

        self.setFrameShape(QFrame.StyledPanel)
        self.resize(100, 100)
        self.layout = QVBoxLayout()
        self.__json_selection_layout()
        self.layout.addSpacing(spacing_between_layouts)
        self.__add_generate_metadata_button()
        self.layout.addSpacing(spacing_between_layouts)

        self.preview_table = QTableView()
        self.layout.addWidget(self.preview_table)

        self.__get_json_file()
        self.__emit_ood_settings_changed()

        self.setLayout(self.layout)

    def ood_settings_changed(self, settings):
        self.settings = settings
        self.__get_json_file()

    def __json_selection_layout(self):
        self.json_file_line = QLineEdit()
        self.json_file_line.setEnabled(False)
        self.layout.addWidget(self.json_file_line)

        self.datasets_combobox = QComboBox()
        self.datasets_combobox.textActivated.connect(self.__on_datasets_combobox_values_change)
        self.layout.addWidget(self.datasets_combobox)

    def __get_json_file(self):
        self.dataset_description_json_file = os.path.join(self.settings.dataset_root_path, "datasets.json")
        if not os.path.isfile(self.dataset_description_json_file):
            return

        self.json_file_line.setText(self.dataset_description_json_file)
        try:
            names = get_datasets_names(self.dataset_description_json_file)
        except KeyError:
            QMessageBox.warning(self, "", "Wrong json format")
            return
        self.datasets_combobox.clear()
        self.datasets_combobox.addItems(names)
        self.datasets_combobox.setCurrentText(self.settings.dataset_name)
        self.settings.set_metadata_folder()

    def __on_datasets_combobox_values_change(self, value):
        self.settings.dataset_name = value
        self.__emit_ood_settings_changed()
        self.generate_button.setEnabled(True)

    def __add_generate_metadata_button(self):
        self.generate_button = QPushButton("Generate metadata")
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.__generate_metadata)
        self.layout.addWidget(self.generate_button)

    def __generate_metadata(self):
        self.generate_button.setEnabled(False)
        try:
            self.settings.set_metadata_folder()
            self.metadata_file = generate_metadata(self.dataset_description_json_file,
                                                   self.settings.dataset_name,
                                                   self.settings.metadata_folder)
            self.__emit_ood_settings_changed()
        except Exception as error:
            print(error)
            QMessageBox.warning(self, "Metadata generation failed", str(error))
        self.generate_button.setEnabled(True)
        self.__preview_pkl_file()

    def __emit_ood_settings_changed(self):
        self.ood_settings_changed_signal.emit(self.settings)

    def __preview_pkl_file(self):
        if os.path.isfile(self.metadata_file):
            df = pd.read_pickle(self.metadata_file)
            if df.shape[0] > 300:
                random_indices = random.sample(range(0, df.shape[0]), 300)
                df = df.iloc[random_indices]
            df[types.LabelsType.name()] = df.apply(lambda row: ', '.join(row[types.LabelsType.name()]), axis=1)
            table_model = TableModel(df.values.tolist(), df.columns.values.tolist())
            self.preview_table.setModel(table_model)
