import os
from pyqtconfig import ConfigManager

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QFileDialog,
    QFrame,
    QLineEdit,
)

from PyQt5.QtCore import pyqtSignal

from tool.pyqt_gui.paths_settings import PathsSettings
from tool.pyqt_gui.qt_utils import helpers


def create_new_session_folder(dataset_root_path):
    existing_sessions = []
    for file in os.listdir(dataset_root_path):
        d = os.path.join(dataset_root_path, file)
        if os.path.isdir(d) and file.startswith("oodsession_"):
            _, session_id = file.split("_", 1)
            try:
                existing_sessions.append(int(session_id))
            except ValueError:
                continue
    next_session = 0
    if len(existing_sessions) > 0:
        existing_sessions.sort()
        next_session = existing_sessions[-1] + 1
    metadata_folder = os.path.join(dataset_root_path,
                                   "oodsession_" + str(next_session))
    os.makedirs(metadata_folder)
    return metadata_folder


class PathsSettingsFrame(QFrame):
    ood_settings_changed_signal = pyqtSignal(PathsSettings)

    def __init__(self, parent, create_new_folder=False):
        super(PathsSettingsFrame, self).__init__(parent)

        self.setting = PathsSettings()
        self.config = ConfigManager(self.setting.get_default_settings(), filename="./ood_config.json")
        self.config.set_defaults(self.config.as_dict())
        self.setting.set_from_config(self.config)
        self.create_new_folder = create_new_folder

        if create_new_folder and os.path.exists(self.setting.dataset_root_path):
            self.setting.metadata_folder = create_new_session_folder(self.setting.dataset_root_path)

        self.setFrameShape(QFrame.StyledPanel)
        self.resize(100, 100)
        self.layout = QVBoxLayout()

        self.database_root_line = QLineEdit()
        self.config.add_handler('dataset_root_path', self.database_root_line)
        helpers.get_dir_layout(self.__get_database_root_dir, self.database_root_line,
                               "Absolute path to dataset folder: ", self.setting.dataset_root_path, self)

        self.working_dir_line = QLineEdit()
        helpers.get_dir_layout(self.__get_working_dir, self.working_dir_line,
                               "Absolute path to working directory: ", self.setting.metadata_folder, self)
        self.setLayout(self.layout)

    def __get_database_root_dir(self):
        self.setting.dataset_root_path = QFileDialog.getExistingDirectory(self, caption='Choose Directory',
                                                                          directory=self.setting.dataset_root_path)
        self.database_root_line.setText(self.setting.dataset_root_path)

        if self.create_new_folder and os.path.exists(self.setting.dataset_root_path):
            self.setting.metadata_folder = create_new_session_folder(self.setting.dataset_root_path)
            self.working_dir_line.setText(self.setting.metadata_folder)

        self.ood_settings_changed_signal.emit(self.setting)
        self.config.save()

    def __get_working_dir(self):
        self.setting.metadata_folder = QFileDialog.getExistingDirectory(self, caption='Choose Working Directory',
                                                                        directory=self.setting.metadata_folder)
        self.ood_settings_changed_signal.emit(self.setting)
        self.working_dir_line.setText(self.setting.metadata_folder)

    def emit_settings(self):
        self.ood_settings_changed_signal.emit(self.setting)
