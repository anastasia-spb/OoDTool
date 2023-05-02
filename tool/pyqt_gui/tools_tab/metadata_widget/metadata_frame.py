import os
import pandas as pd
import random

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QFrame,
    QLineEdit,
    QComboBox, QMessageBox, QTableView,
)

from PyQt5.QtCore import Qt, QAbstractTableModel, pyqtSignal

from tool.core.metadata_generator.generator import generate_metadata
from tool.pyqt_gui.paths_settings import PathsSettings
from tool.core.data_types import types


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
    ood_settings_changed_signal = pyqtSignal(PathsSettings)

    def __init__(self, parent):
        super(MetadataFrame, self).__init__(parent)

        self.settings = PathsSettings()
        self.metadata_file = None
        self.dataset_name = ''

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
        if not os.path.exists(self.settings.dataset_root_path):
            return

        description_files = []
        for file in os.listdir(self.settings.dataset_root_path):
            if file.endswith(".json"):
                description_files.append(file)

        if len(description_files) == 0:
            return

        self.datasets_combobox.clear()
        self.datasets_combobox.addItems(description_files)
        self.datasets_combobox.setCurrentText(self.dataset_name)

        self.dataset_description_json_file = os.path.join(self.settings.dataset_root_path, description_files[0])
        if not os.path.isfile(self.dataset_description_json_file):
            return

        self.json_file_line.setText(self.dataset_description_json_file)

    def __on_datasets_combobox_values_change(self, value):
        self.dataset_name = value
        self.generate_button.setEnabled(True)

    def __add_generate_metadata_button(self):
        self.generate_button = QPushButton("Generate metadata")
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.__generate_metadata)
        self.layout.addWidget(self.generate_button)

    def __generate_metadata(self):
        self.generate_button.setEnabled(False)
        try:
            self.metadata_file = generate_metadata(self.dataset_description_json_file,
                                                   self.settings.metadata_folder)
        except Exception as error:
            print(error)
            QMessageBox.warning(self, "Metadata generation failed", str(error))
        self.generate_button.setEnabled(True)
        self.__preview_pkl_file()

    def __preview_pkl_file(self):
        if os.path.isfile(self.metadata_file):
            df = pd.read_pickle(self.metadata_file)
            if df.shape[0] > 300:
                random_indices = random.sample(range(0, df.shape[0]), 300)
                df = df.iloc[random_indices]
            df[types.LabelsType.name()] = df.apply(lambda row: ', '.join(row[types.LabelsType.name()]), axis=1)
            table_model = TableModel(df.values.tolist(), df.columns.values.tolist())
            self.preview_table.setModel(table_model)
