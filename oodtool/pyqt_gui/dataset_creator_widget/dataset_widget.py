from PyQt5.QtWidgets import (
    QMainWindow,
)

from oodtool.pyqt_gui.dataset_creator_widget.create_dataset_description import create_dataset_description

from oodtool.pyqt_gui.dataset_creator_widget.file_dialog_widget import FileDiaolgWidget


class DatasetWindow(QMainWindow):
    def __init__(self, metadata_dir: str, data_dir: str):
        super(DatasetWindow, self).__init__()

        self.main_widget = FileDiaolgWidget(
            root_path=data_dir, parent=self
        )
        self.main_widget.update_parameters_dock_widget.connect(
            self._update_input_line_text
        )

        self.metadata_dir = metadata_dir
        self.data_dir = data_dir

        self.setCentralWidget(self.main_widget)
        self.setWindowTitle("Create dataset")
        self.main_widget.exec()

    def _update_input_line_text(self, values):
        train_folders = values[0]
        test_folders = values[1]
        create_dataset_description(train_folders, test_folders, self.metadata_dir, self.data_dir)
