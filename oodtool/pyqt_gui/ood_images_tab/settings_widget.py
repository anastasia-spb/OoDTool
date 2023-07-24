import pandas as pd
import os
import logging

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFileDialog,
    QFrame,
    QLineEdit,
    QMessageBox,
)

from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal, QTimer

from pyqtconfig import ConfigManager

from oodtool.pyqt_gui.qt_utils import helpers
from oodtool.pyqt_gui.data_loader.loader import DataLoader

from oodtool.pyqt_gui.ood_images_tab.filters_frame import FiltersFrame, FilterSettings
from oodtool.pyqt_gui.qt_utils.create_new_session_folder import create_new_session_folder
from oodtool.pyqt_gui.dataset_creator_widget.dataset_widget import DatasetWindow
from oodtool.pyqt_gui.qt_utils.helpers import ShowImageFrame
from oodtool.pyqt_gui.plot_window.plot_window import PlotWindow
from oodtool.pyqt_gui.qt_utils.qt_types import ImageInfo


def get_default_settings():
    return {
        "dataset_root_path": "",
        "metadata_folder": "",
    }


class PathsSettingsFrame(QFrame):
    def __init__(self, parent):
        super(PathsSettingsFrame, self).__init__(parent)

        self.dataset_root_path = ''
        self.metadata_folder = ''

        self.config = ConfigManager(get_default_settings(), filename="./ood_config.json")
        self.config.set_defaults(self.config.as_dict())
        self.dataset_root_path = self.config.get('dataset_root_path')
        self.metadata_folder = self.config.get('metadata_folder')

        self.setFrameShape(QFrame.StyledPanel)
        self.layout = QVBoxLayout()

        self.database_root_line = QLineEdit()
        self.config.add_handler('dataset_root_path', self.database_root_line)
        helpers.get_dir_layout(self.__get_database_root_dir, self.database_root_line,
                               "Absolute path to dataset folder: ", self.dataset_root_path, self)

        self.working_dir_line = QLineEdit()
        self.config.add_handler('metadata_folder', self.working_dir_line)
        helpers.get_dir_layout(self.__get_working_dir, self.working_dir_line,
                               "Absolute path to working directory: ", self.metadata_folder, self)
        self.setLayout(self.layout)

    def __get_database_root_dir(self):
        self.dataset_root_path = QFileDialog.getExistingDirectory(self, caption='Choose Directory',
                                                                  directory=self.dataset_root_path)
        self.database_root_line.setText(self.dataset_root_path)
        self.config.save()

    def __get_working_dir(self):
        self.metadata_folder = QFileDialog.getExistingDirectory(self, caption='Choose Working Directory',
                                                                directory=self.metadata_folder)
        self.working_dir_line.setText(self.metadata_folder)
        self.config.save()

    def get_dataset_root_path(self):
        return self.dataset_root_path

    def get_metadata_folder(self):
        return self.metadata_folder

    def set_metadata_folder(self, folder: str):
        self.metadata_folder = folder
        self.working_dir_line.setText(self.metadata_folder)
        self.config.save()


class SettingsWidget(QWidget):
    _images_signal = pyqtSignal(list)
    _filter_settings = pyqtSignal(FilterSettings)
    _labels_signal = pyqtSignal(list)
    _show_neighbours_signal = pyqtSignal(ImageInfo)

    def __init__(self, parent, data_loader: DataLoader):
        super(SettingsWidget, self).__init__(parent)

        self.layout = QVBoxLayout()

        # Keep track of plot window, only one instance allowed
        self.plot_window = None
        self.ood_window = None
        self.neighbours_window = None
        self.projector_window = None
        self.legend = None
        # @TODO: Head shall depend on dataset
        self.head_idx = 0

        self.data_loader = data_loader

        # Load Data
        load_data_container = QFrame()
        load_data_container.setFrameShape(QFrame.StyledPanel)
        container_layout = QVBoxLayout(load_data_container)

        self.path_settings_frame = PathsSettingsFrame(self)
        self.path_settings_frame.setMaximumHeight(200)
        container_layout.addWidget(self.path_settings_frame, alignment=Qt.AlignmentFlag.AlignTop)

        load_buttons_layout = QHBoxLayout()
        self.load_button = QPushButton("Load data")
        self.load_button.clicked.connect(lambda: self.__load_data())
        load_buttons_layout.addWidget(self.load_button, alignment=Qt.AlignmentFlag.AlignTop)

        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(lambda: self.__load_data(reload=True))
        load_buttons_layout.addWidget(self.reload_button, alignment=Qt.AlignmentFlag.AlignTop)

        container_layout.addLayout(load_buttons_layout)
        self.layout.addWidget(load_data_container)

        # Time for Data Loading and Image Grid update
        self.timer = QTimer(self, interval=500)  # time in ms
        self.timer.timeout.connect(self.__enable_load_buttons)

        # Image preview frame
        image_preview_container = QFrame()
        image_preview_container.setFrameShape(QFrame.Panel)
        image_preview_container_layout = QVBoxLayout(image_preview_container)

        self.image_preview_widget = ShowImageFrame(self)
        image_preview_container_layout.addWidget(self.image_preview_widget)

        self.layout.addWidget(image_preview_container)

        # Filters container
        filters_container = QFrame()
        filters_container.setFrameShape(QFrame.Panel)
        filter_container_layout = QVBoxLayout(filters_container)

        self.filters_frame = FiltersFrame(self)
        filter_container_layout.addWidget(self.filters_frame, alignment=Qt.AlignmentFlag.AlignTop)

        filter_button = QPushButton("Apply")
        filter_button.clicked.connect(self.__filter_images)
        filter_container_layout.addWidget(filter_button)

        self.layout.addWidget(filters_container)

        self.setLayout(self.layout)

    def __enable_load_buttons(self):
        self.reload_button.setDisabled(False)
        self.load_button.setDisabled(False)

    def show_img_meta(self, img_meta):
        self.image_preview_widget.show_image(img_meta)

    def __filter_images(self):
        self._images_signal.emit(self.data_loader.get_images_filtered_info(self.filters_frame.get_settings()))
        if self.plot_window is not None:
            self.plot_window.reload(self.data_loader, self.legend, self.filters_frame.get_settings())

    def __select_metadata_folder(self):
        msg_box = QMessageBox()

        msg_box.setObjectName("SessionTypeSelect")
        msg_box.setText("Do you want to create new session folder or use the current one?")
        msg_box.addButton(QPushButton('Create new session'), QMessageBox.YesRole)
        msg_box.addButton(QPushButton('Use the current session'), QMessageBox.NoRole)
        ret = msg_box.exec_()

        if ret == 0:
            if not os.path.exists(self.path_settings_frame.get_dataset_root_path()):
                logging.info("New session creation requested. Dataset root path doesn't exist.")
                msg_box.close()
                return

            metadata_folder = create_new_session_folder(self.path_settings_frame.get_dataset_root_path())
            self.path_settings_frame.set_metadata_folder(metadata_folder)
            self.__create_dataset()
            msg_box.close()
            return

        msg_box.close()

    def __create_dataset(self):
        file_dialog_window = DatasetWindow(data_dir=self.path_settings_frame.get_dataset_root_path(),
                                           metadata_dir=self.path_settings_frame.get_metadata_folder())
        file_dialog_window.show()

    def __close_all_tool_window(self):
        if self.plot_window is not None:
            self.plot_window.close()
            self.plot_window = None

        if self.ood_window is not None:
            self.ood_window.close()
            self.ood_window = None

        if self.neighbours_window is not None:
            self.neighbours_window.close()
            self.neighbours_window = None

        if self.projector_window is not None:
            self.projector_window.close()
            self.projector_window = None

    def __load_data(self, reload=False):

        self.reload_button.setDisabled(True)
        self.load_button.setDisabled(True)

        self.__close_all_tool_window()

        if not reload:
            self.__select_metadata_folder()

        metadata_folder = self.path_settings_frame.get_metadata_folder()
        dataset_root_path = self.path_settings_frame.get_dataset_root_path()

        self.data_loader.load_data(dataset_root_path, metadata_folder)
        if not self.data_loader.status_loaded:
            logging.warning("No metadata found in the selected session session %s.", metadata_folder)
            return

        self._labels_signal.emit(self.data_loader.get_labels())
        self.filters_frame.save_labels(self.data_loader.get_labels())
        self.filters_frame.update_ood_methods(self.data_loader.get_available_ood_methods())
        self._images_signal.emit(self.data_loader.get_images_filtered_info(self.filters_frame.get_settings()))

        self.timer.start()

    def export_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,
                                                  "Save File", "", "Text Files(*.csv)",
                                                  options=options)
        if fileName:
            images_list = \
                self.data_loader.get_images_filtered_info(self.filters_frame.get_settings())
            images_list_df = pd.DataFrame.from_records([s.to_dict() for s in images_list])
            images_list_df.to_csv(fileName)

    def open_ood_window(self):
        if self.ood_window is None:
            from oodtool.pyqt_gui.ood_widget.get_ood_score_window import OoDScoreWindow
            self.ood_window = OoDScoreWindow(self, self.path_settings_frame.get_metadata_folder(),
                                             self.path_settings_frame.get_dataset_root_path(), head_idx=self.head_idx)
        self.ood_window.show()
        self.ood_window.activateWindow()
        self.ood_window.raise_()

    def open_neighbours_window(self):
        if self.neighbours_window is None:
            from oodtool.pyqt_gui.tools.distance_widget.distance_widget import NeigboursWindow
            self.neighbours_window = NeigboursWindow(self, self.path_settings_frame.get_metadata_folder())
        self.neighbours_window.show()
        self.neighbours_window.activateWindow()
        self.neighbours_window.raise_()

    def open_projector_window(self):
        if self.projector_window is None:
            from oodtool.pyqt_gui.tools.projector_widget.projector_widget import ProjectorWindow
            self.projector_window = ProjectorWindow(self, self.path_settings_frame.get_metadata_folder())
        self.projector_window.show()
        self.projector_window.activateWindow()
        self.projector_window.raise_()

    def open_plot_window(self):
        if not self.data_loader.status_loaded:
            return

        if self.plot_window is None:
            self.plot_window = PlotWindow(self, self.data_loader, self.legend, self.filters_frame.get_settings(),
                                          self.head_idx)
            # Show image info on pick
            self.plot_window.image_path_signal.connect(self.show_img_meta)
            self.plot_window.image_path_signal.connect(self.show_neighbours)
        self.plot_window.show()
        self.plot_window.activateWindow()
        self.plot_window.raise_()

    def show_neighbours(self, img_meta):
        self._show_neighbours_signal.emit(img_meta)

    def legend_updated(self, legend):
        self.legend = legend
