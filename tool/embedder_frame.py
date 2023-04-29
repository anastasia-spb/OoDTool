import os

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QFrame,
    QScrollArea,
    QWidget
)

from PyQt5.QtCore import Qt, pyqtSignal

from tool.embedder_frame_widgets import embedder_widget, projector_widget
from tool.ood_settings import OoDSettings


class EmbeddersWidget(QWidget):
    def __init__(self, parent):
        super(EmbeddersWidget, self).__init__(parent)

        self.settings = OoDSettings()
        self.uidentifier = 0

        self.layout = QVBoxLayout()

        self.embedder_widget = embedder_widget.EmbedderFrame(self, self.settings)
        self.layout.addWidget(self.embedder_widget, alignment=Qt.AlignmentFlag.AlignTop)

        self.projector_widget = projector_widget.ProjectorFrame(self, self.settings)
        self.layout.addWidget(self.projector_widget, alignment=Qt.AlignmentFlag.AlignTop)

        self.layout.addStretch()
        self.setLayout(self.layout)

    def ood_settings_changed(self, settings):
        self.settings = settings
        self.embedder_widget.ood_settings_changed(settings)
        self.projector_widget.ood_settings_changed(settings)


class EmbeddersFrame(QFrame):
    send_embeddings_files_paths_signal = pyqtSignal(list)

    def __init__(self, parent):
        super(EmbeddersFrame, self).__init__(parent)

        self.settings = OoDSettings()

        self.setFrameShape(QFrame.NoFrame)
        self.resize(100, 100)
        self.layout = QVBoxLayout()
        self.embedders_widget = EmbeddersWidget(self)

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.embedders_widget)

        self.layout.addWidget(self.scroll)
        self.setLayout(self.layout)

    def ood_settings_changed(self, settings):
        self.settings = settings
        self.embedders_widget.ood_settings_changed(settings)

