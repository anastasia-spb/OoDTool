import os

from PyQt5.QtWidgets import (
    QHBoxLayout,
    QFrame,
    QLineEdit,
    QCheckBox, QLabel
)

from PyQt5.QtCore import pyqtSignal


class AdditionalSettings:
    def __init__(self):
        self.use_gt_labels = True
        self.ood_threshold = 0.7


class AnalyzeTabSettingsFrame(QFrame):
    analyze_settings_changed_signal = pyqtSignal(AdditionalSettings)

    def __init__(self, parent):
        super(AnalyzeTabSettingsFrame, self).__init__(parent)

        self.setting = AdditionalSettings()

        self.setFrameShape(QFrame.StyledPanel)

        self.layout = QHBoxLayout()

        self.use_gt = QCheckBox("Use GT classification")
        self.use_gt.setChecked(self.setting.use_gt_labels)
        self.use_gt.stateChanged.connect(lambda: self.__use_gt_state())
        self.layout.addWidget(self.use_gt)

        self.layout.addSpacing(20)

        text = QLabel("Select OoD Threshold", self)
        self.layout.addWidget(text)

        self.sl = QLineEdit(str(self.setting.ood_threshold))
        self.layout.addWidget(self.sl)
        self.sl.returnPressed.connect(self.__ood_threshold_changed)

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

    def emit_settings(self):
        self.analyze_settings_changed_signal.emit(self.setting)
