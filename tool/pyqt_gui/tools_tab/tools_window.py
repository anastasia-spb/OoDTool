from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QHBoxLayout, QLabel,
)

from PyQt5.QtGui import QPixmap

from tool.pyqt_gui.tools_tab.metadata_widget.metadata_frame import MetadataFrame
from tool.pyqt_gui.tools_tab.embedder_widget.embedder_frame import EmbedderFrame
from tool.pyqt_gui.tools_tab.projector_widget.projector_widget import ProjectorFrame
from tool.pyqt_gui.paths_settings_frame import PathsSettingsFrame
from tool.pyqt_gui.tools_tab.distance_widget.distance_widget import DistanceFrame


class LogoFrame(QWidget):
    def __init__(self, parent):
        super(LogoFrame, self).__init__(parent)

        self.layout = QHBoxLayout()

        label = QLabel(self)
        pixmap = QPixmap('tool/pyqt_gui/gui_graphics/ood_logo_v3_small.png')
        pixmap.scaledToHeight(60)
        label.setPixmap(pixmap)
        label.resize(pixmap.width(),
                     pixmap.height())
        self.layout.addWidget(label)

        self.setLayout(self.layout)


class ToolsWindow(QWidget):
    def __init__(self, parent):
        super(ToolsWindow, self).__init__(parent)

        space_between_frames = 10

        self.layout = QVBoxLayout()

        self.upper_layout = QHBoxLayout()
        logo_frame = LogoFrame(self)
        self.upper_layout.addWidget(logo_frame)

        self.common_settings_frame = PathsSettingsFrame(self, create_new_folder=True)
        self.upper_layout.addWidget(self.common_settings_frame)
        self.layout.addLayout(self.upper_layout)

        self.bottom_layout = QHBoxLayout()

        self.widget_1 = MetadataFrame(self)
        self.bottom_layout.addWidget(self.widget_1)
        self.bottom_layout.addSpacing(space_between_frames)

        self.widget_2 = EmbedderFrame(self)
        self.bottom_layout.addWidget(self.widget_2)
        self.bottom_layout.addSpacing(space_between_frames)

        self.bottom_right_layout = QVBoxLayout()
        self.projector_frame = ProjectorFrame(self)
        self.bottom_right_layout.addWidget(self.projector_frame)

        self.distance_frame = DistanceFrame(self)
        self.bottom_right_layout.addWidget(self.distance_frame)

        self.bottom_layout.addLayout(self.bottom_right_layout)
        self.layout.addLayout(self.bottom_layout)

        # Add subscription of all widgets to common setting
        self.common_settings_frame.ood_settings_changed_signal.connect(self.widget_1.ood_settings_changed)
        self.common_settings_frame.ood_settings_changed_signal.connect(self.widget_2.ood_settings_changed)
        self.common_settings_frame.ood_settings_changed_signal.connect(self.projector_frame.ood_settings_changed)
        self.common_settings_frame.ood_settings_changed_signal.connect(self.distance_frame.ood_settings_changed)

        self.common_settings_frame.emit_settings()

        self.setLayout(self.layout)
