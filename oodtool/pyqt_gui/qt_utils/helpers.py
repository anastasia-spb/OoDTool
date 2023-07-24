import os

import pandas as pd

from PyQt5.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QGridLayout,
    QFrame,
    QTextEdit, QLabel, QLineEdit, QApplication, QMenu, QWidget
)

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QIcon

from oodtool.pyqt_gui.qt_utils.qt_types import ImageInfo


def get_dir_layout(callback_fn, output_dir_line, label, default_dir, parent):
    text = QLabel(label, parent)
    parent.layout.addWidget(text)

    output_dir_layout = QHBoxLayout()
    select_dir_button = QPushButton()
    select_dir_button.setIcon(QIcon("oodtool/pyqt_gui/gui_graphics/select.png"))

    output_dir_line.setEnabled(False)
    output_dir_line.setText(default_dir)
    select_dir_button.clicked.connect(callback_fn)
    output_dir_layout.addWidget(output_dir_line)
    output_dir_layout.addWidget(select_dir_button)
    parent.layout.addLayout(output_dir_layout)


class ExportToCsvFrame(QFrame):
    def __init__(self, parent, current_img_info: ImageInfo):
        super(ExportToCsvFrame, self).__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)

        if current_img_info is None:
            return

        self.layout = QVBoxLayout()
        self.parent = parent

        grid_layout = QGridLayout()
        text_key = QLabel("key", self)
        grid_layout.addWidget(text_key, 0, 0)

        text_label = QLabel("value", self)
        grid_layout.addWidget(text_label, 0, 1)

        self.current_img_info = current_img_info

        self.__prepare_export_info(grid_layout)
        self.layout.addLayout(grid_layout)

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(lambda: self.__export_to_csv())
        self.layout.addWidget(self.export_button, alignment=Qt.AlignmentFlag.AlignBottom)

        self.setLayout(self.layout)

    def __prepare_export_info(self, grid_layout):
        self.keys = []
        self.lines = []

        info_dict = self.current_img_info.to_dict()
        idx = 0
        for key, value in info_dict.items():
            key_line = QLineEdit(key)
            self.keys.append(key_line)
            grid_layout.addWidget(self.keys[-1], idx, 0)

            value_line = QLineEdit(str(value))
            self.lines.append(value_line)
            grid_layout.addWidget(self.lines[-1], idx, 1)
            idx += 1

        # Add additional lines
        for i in range(idx, idx + 3):
            notes_line = QLineEdit('')
            self.keys.append(notes_line)
            grid_layout.addWidget(self.keys[-1], i, 0)

            value_line = QLineEdit('')
            self.lines.append(value_line)
            grid_layout.addWidget(self.lines[-1], i, 1)

    def __export_to_csv(self):
        if self.current_img_info is None:
            return

        _, filename = os.path.split(self.current_img_info.relative_path)

        df = dict()

        for key, val in zip(self.keys, self.lines):
            k = key.text()
            if k != "":
                df[key.text()] = val.text()

        csv_df = pd.DataFrame(df, index=[0])

        export_folder = os.path.join(self.current_img_info.metadata_dir, 'ood_export')
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)

        csv_filename = "".join((filename, ".csv"))
        csv_df.to_csv(os.path.join(export_folder, csv_filename))

        self.parent.close_window()


class ExportToCsvWindow(QMainWindow):
    def __init__(self, parent=None, current_img_info: ImageInfo = None):
        super(ExportToCsvWindow, self).__init__(parent)

        self.central_widget = ExportToCsvFrame(self, current_img_info)
        self.setCentralWidget(self.central_widget)
        self.setWindowTitle("Export...")

    def close_window(self):
        self.close()


class ShowImageFrame(QWidget):
    def __init__(self, parent):
        super(ShowImageFrame, self).__init__(parent)

        self.layout = QHBoxLayout()

        self.left_layout = QVBoxLayout()
        self.img_meta = QTextEdit()
        self.img_meta.setMaximumHeight(150)
        self.img_meta.setReadOnly(True)
        self.left_layout.addWidget(self.img_meta)

        self.image_menu = None
        self.image_label = QLabel(self)
        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.left_layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_label.customContextMenuRequested.connect(lambda position: self.__open_image_menu(position))

        self.layout.addLayout(self.left_layout)

        self.current_img_info = None
        self.pix = None
        self.setLayout(self.layout)

        self.__create_image_menu_actions()

    def __create_image_menu_actions(self):
        self.image_menu = QMenu()
        self.export_action = self.image_menu.addAction("&Export")

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        shift_pressed = (modifiers == Qt.ShiftModifier)

        if shift_pressed:
            if event.angleDelta().y() > 0:
                self.__scale_image(True)
            elif event.angleDelta().y() < 0:
                self.__scale_image(False)

    def __open_image_menu(self, position):
        if self.image_menu is None:
            return

        pix_pos = self.mapToGlobal(self.image_label.pos())
        click_pos = QPoint(pix_pos.x() + position.x(), pix_pos.y() + position.y())
        action = self.image_menu.exec_(click_pos)
        if action == self.export_action:
            self.__export_to_csv()

    def __scale_image(self, up: bool):
        if self.pix is None:
            return

        h = self.pix.height()

        if up:
            max_h = 600
            h = min(max_h, h + 20)
        else:
            min_h = 40
            h = max(min_h, h - 20)

        self.pix = self.pix.scaledToHeight(h)
        self.image_label.setPixmap(self.pix)

    def show_image(self, info: ImageInfo):
        self.current_img_info = info

        self.img_meta.clear()
        self.img_meta.setText(info.relative_path)
        self.img_meta.append("OoD Score: {:10.3f}".format(info.ood_score))
        if info.confidence is not None:
            strings = ['{:.3f}'.format(x) for x in info.confidence]
            self.img_meta.append("Predicted class: " + ', '.join(info.predicted_label))
            self.img_meta.append("Confidence: " + ", ".join(strings))

        self.img_meta.append("GT Label: " + info.gt_label)

        self.pix = QPixmap(os.path.join(info.dataset_root_dir, info.relative_path))
        # Set minimum height
        self.pix = self.pix.scaledToHeight(250)
        self.image_label.setPixmap(self.pix)

    def __export_to_csv(self):
        export_window = ExportToCsvWindow(self, self.current_img_info)
        export_window.show()
