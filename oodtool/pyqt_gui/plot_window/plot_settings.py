import os

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QFrame,
    QLineEdit,
    QCheckBox, QLabel
)

from PyQt5.QtCore import pyqtSignal


class AdditionalSettings:
    def __init__(self):
        self.ood_threshold = 0.7
        self.highlight_test_samples = False


class AnalyzeTabSettingsFrame(QFrame):
    analyze_settings_changed_signal = pyqtSignal(AdditionalSettings)

    def __init__(self, parent):
        super(AnalyzeTabSettingsFrame, self).__init__(parent)

        self.setting = AdditionalSettings()

        self.setFrameShape(QFrame.StyledPanel)

        self.layout = QVBoxLayout()

        text = QLabel("Select OoD Threshold", self)
        self.layout.addWidget(text)

        self.sl = QLineEdit(str(self.setting.ood_threshold))
        self.layout.addWidget(self.sl)
        self.sl.returnPressed.connect(self.__ood_threshold_changed)

        self.highlight_box = QCheckBox("Highlight test samples")
        self.highlight_box.setChecked(False)
        self.highlight_box.stateChanged.connect(lambda: self.__check_box_state_changed())
        self.layout.addWidget(self.highlight_box)

        self.layout.addStretch()
        self.setLayout(self.layout)

    def __check_box_state_changed(self):
        self.setting.highlight_test_samples = self.highlight_box.isChecked()
        self.analyze_settings_changed_signal.emit(self.setting)

    def __ood_threshold_changed(self):
        t = self.sl.text()
        try:
            threshold = float(t)
        except ValueError:
            threshold = 0.0

        self.setting.ood_threshold = max(0.0, min(threshold, 1.0))
        self.analyze_settings_changed_signal.emit(self.setting)

    def emit_settings(self):
        self.analyze_settings_changed_signal.emit(self.setting)
