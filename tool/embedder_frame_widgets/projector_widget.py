import os
import pandas as pd
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QFrame,
    QLineEdit,
    QComboBox,
    QLabel
)

from tool.data_projectors.data_projector import DataProjector
from tool.data_types import types
from tool.ood_settings import OoDSettings
from tool.qt_utils import find_pkl


class ProjectorFrame(QFrame):

    def __init__(self, parent, settings: OoDSettings):
        super(ProjectorFrame, self).__init__(parent)

        spacing_between_layouts = 30
        self.settings = settings

        self.output_file = ''
        self.selected_method = DataProjector.method_name

        self.setFrameShape(QFrame.StyledPanel)
        self.resize(100, 100)
        self.layout = QVBoxLayout()

        text = QLabel("Projector", self)
        font = text.font()
        font.setBold(True)
        text.setFont(font)
        self.layout.addWidget(text)

        self.__embeddings_selection_layout()
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

    def __embeddings_selection_layout(self):
        self.datasets_combobox = QComboBox()
        self.datasets_combobox.textActivated.connect(self.__on_datasets_combobox_values_change)
        self.datasets_combobox.addItems(DataProjector.methods)
        self.layout.addWidget(self.datasets_combobox)

    def __on_datasets_combobox_values_change(self, value):
        self.selected_method = value

    def __add_generate_metadata_button(self):
        self.generate_button = QPushButton("Fit")
        self.generate_button.setEnabled(True)
        self.generate_button.clicked.connect(self.__fit_embeddings)
        self.layout.addWidget(self.generate_button)

    def __fit_embeddings(self):
        self.generate_button.setEnabled(False)
        projector = DataProjector(self.selected_method)
        if projector is None:
            return

        data_df = pd.read_pickle(find_pkl.get_embeddings_file(self.settings.metadata_folder))
        # TODO: drop embeddings
        X = data_df[types.EmbeddingsType.name()].tolist()
        embeddings = np.array(X, dtype=np.dtype('float64'))
        x = projector.fit_transform(embeddings)

        data_df[types.ProjectedEmbeddingsType.name()] = x.tolist()

        timestamp_str = datetime.now().strftime("%y%m%d_%H%M%S")
        name = "".join((self.selected_method, "_", timestamp_str, ".2emb.pkl"))
        self.output_file = os.path.join(self.settings.metadata_folder, name)
        data_df.to_pickle(self.output_file)
        self.output_file_line.setText(self.output_file)
        self.generate_button.setEnabled(True)
