import os

from pyqtconfig import ConfigManager

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QFileDialog,
    QFrame,
    QLineEdit,
    QCheckBox,
    QLabel,
)

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal

from tool.ood_settings import OoDSettings
from tool.qt_utils import helpers


class OoDSettingsFrame(QFrame):
    ood_settings_changed_signal = pyqtSignal(OoDSettings)

    def __init__(self, parent):
        super(OoDSettingsFrame, self).__init__(parent)

        self.setting = OoDSettings()
        self.config = ConfigManager(self.setting.get_default_settings(), filename="./ood_config.json")
        self.config.set_defaults(self.config.as_dict())

        self.setting.set_from_config(self.config)

        self.setFrameShape(QFrame.StyledPanel)
        self.resize(100, 100)
        self.layout = QVBoxLayout()

        self.__add_logo()

        self.database_root_line = QLineEdit()
        self.config.add_handler('dataset_root_path', self.database_root_line)
        helpers.get_dir_layout(self.__get_database_root_dir, self.database_root_line,
                               "Absolute path to dataset folder: ", self.setting.dataset_root_path, self)

        self.working_dir_line = QLineEdit()
        self.config.add_handler('working_dir', self.working_dir_line)
        helpers.get_dir_layout(self.__get_working_dir, self.working_dir_line,
                               "Absolute path to working directory: ", self.setting.working_dir, self)

        self.__use_cuda_box()
        self.setLayout(self.layout)

    def __add_logo(self):
        label = QLabel(self)
        pixmap = QPixmap('tool/gui_graphics/ood_logo_v3_small.png')
        pixmap.scaledToHeight(60)
        label.setPixmap(pixmap)
        label.resize(pixmap.width(),
                     pixmap.height())
        self.layout.addWidget(label)

    def __get_database_root_dir(self):
        self.setting.dataset_root_path = QFileDialog.getExistingDirectory(self, caption='Choose Directory',
                                                                          directory=self.setting.dataset_root_path)
        self.database_root_line.setText(self.setting.dataset_root_path)
        self.ood_settings_changed_signal.emit(self.setting)
        self.config.save()

    def __get_working_dir(self):
        self.setting.working_dir = QFileDialog.getExistingDirectory(self, caption='Choose Working Directory',
                                                                    directory=self.setting.working_dir)
        self.setting.set_metadata_folder()
        self.working_dir_line.setText(self.setting.working_dir)
        self.ood_settings_changed_signal.emit(self.setting)
        self.config.save()

    def __use_cuda_box(self):
        self.use_cuda_box = QCheckBox("Use cuda")
        self.use_cuda_box.setChecked(self.setting.use_cuda)
        self.use_cuda_box.stateChanged.connect(lambda: self.__use_cuda_box_state())
        self.layout.addWidget(self.use_cuda_box)

    def __use_cuda_box_state(self):
        self.setting.use_cuda = self.use_cuda_box.isChecked()
        self.ood_settings_changed_signal.emit(self.setting)

    def ood_settings_changed(self, settings):
        self.setting = settings
        self.database_root_line.setText(self.setting.dataset_root_path)
        self.ood_settings_changed_signal.emit(self.setting)

    def emit_settings(self):
        self.ood_settings_changed_signal.emit(self.setting)
