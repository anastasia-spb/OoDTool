import os

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QFrame,
    QLineEdit,
    QCheckBox, QSlider, QLabel
)

from PyQt5.QtCore import pyqtSignal, Qt
from pyqtconfig import ConfigManager

from tool.qt_utils import helpers


class AnalyzeTabSettings:
    def __init__(self):
        self.dataset_root_path = ''
        self.metadata_dir = ''
        self.use_gt_labels = True
        self.ood_threshold = 0.7

    def set_from_config(self, config):
        self.dataset_root_path = config.get('dataset_root_path')

    def get_default_settings(self):
        return {
            "dataset_root_path": self.dataset_root_path,
        }


class AnalyzeTabSettingsFrame(QFrame):
    analyze_settings_changed_signal = pyqtSignal(AnalyzeTabSettings)

    def __init__(self, parent):
        super(AnalyzeTabSettingsFrame, self).__init__(parent)

        self.setting = AnalyzeTabSettings()
        self.config = ConfigManager(self.setting.get_default_settings(), filename="./ood_config.json")
        self.config.set_defaults(self.config.as_dict())

        self.setting.set_from_config(self.config)

        self.setFrameShape(QFrame.StyledPanel)
        self.resize(100, 100)
        self.layout = QVBoxLayout()

        self.database_root_line = QLineEdit()
        helpers.get_dir_layout(self.__get_database_root_dir, self.database_root_line,
                               "Absolute path to dataset folder: ", self.setting.dataset_root_path, self)
        self.config.add_handler('dataset_root_path', self.database_root_line)

        self.metadata_dir_line = QLineEdit()
        helpers.get_dir_layout(self.__get_metadata_dir, self.metadata_dir_line,
                               "Absolute path to metadata folder: ", self.setting.metadata_dir, self)

        self.h_layout = QHBoxLayout()

        self.use_gt = QCheckBox("Use GT classification")
        self.use_gt.setChecked(self.setting.use_gt_labels)
        self.use_gt.stateChanged.connect(lambda: self.__use_gt_state())
        self.h_layout.addWidget(self.use_gt)

        self.h_layout.addSpacing(20)

        text = QLabel("Select OoD Threshold", self)
        self.h_layout.addWidget(text)

        self.sl = QLineEdit(str(self.setting.ood_threshold))
        self.h_layout.addWidget(self.sl)
        self.sl.returnPressed.connect(self.__ood_threshold_changed)

        self.layout.addLayout(self.h_layout)

        self.layout.addStretch()
        self.setLayout(self.layout)

    def __ood_threshold_changed(self):
        t = self.sl.text()
        try:
            threshold = float(t)
        except ValueError:
            threshold = 0.0

        self.setting.ood_threshold = max(0.0, min(threshold, 1.0))
        self.analyze_settings_changed_signal.emit(self.setting)

    def __use_gt_state(self):
        self.setting.use_gt_labels = self.use_gt.isChecked()
        self.analyze_settings_changed_signal.emit(self.setting)

    def __get_database_root_dir(self):
        self.setting.dataset_root_path = QFileDialog.getExistingDirectory(self, caption='Choose Directory',
                                                                          directory=os.getcwd())
        self.database_root_line.setText(self.setting.dataset_root_path)
        self.config.save()
        self.analyze_settings_changed_signal.emit(self.setting)

    def __get_metadata_dir(self):
        self.setting.metadata_dir = QFileDialog.getExistingDirectory(self, caption='Choose Directory',
                                                                     directory=os.getcwd())
        self.metadata_dir_line.setText(self.setting.metadata_dir)
        self.analyze_settings_changed_signal.emit(self.setting)
