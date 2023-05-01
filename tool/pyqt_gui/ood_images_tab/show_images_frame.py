import os
import pandas as pd
import random
import math
from typing import List

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
)
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QFrame,
    QScrollArea, QLabel, QHBoxLayout, QGridLayout
)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from tool.pyqt_gui.analyze_settings import AdditionalSettings
from tool.pyqt_gui.paths_settings import PathsSettings
from tool.core.data_types import types
from tool.pyqt_gui.qt_utils import find_pkl
from tool.pyqt_gui.qt_utils.images_grid import ImagesGrid
from tool.pyqt_gui.qt_utils.qt_types import ImageInfo


def get_images_to_show(ood_file_path: str, ascending: bool, labels: List[str], absolute_path_to_dataset: str,
                       metadata_dir: str, ood_threshold, with_high_confidence=False, embeddings_df=None,
                       use_gt_labels=True):
    images_info = []
    ood_df = pd.read_pickle(ood_file_path)

    def create_gt_conf(label):
        conf = [0.0] * len(labels)
        conf[labels.index(label)] = 1.0
        return conf

    if embeddings_df is not None:
        ood_df = pd.merge(ood_df, embeddings_df[[types.RelativePathType.name(),
                                                 types.ClassProbabilitiesType.name(),
                                                 types.LabelType.name()]],
                          on=types.RelativePathType.name(), how='inner')

        if use_gt_labels:
            ood_df["confidence"] = ood_df[types.LabelType.name()].apply(lambda label: create_gt_conf(label))
        else:
            ood_df["confidence"] = ood_df[types.ClassProbabilitiesType.name()]
    else:
        ood_df["confidence"] = [0.0]

    if with_high_confidence:
        ood_df = ood_df[ood_df[types.OoDScoreType.name()] > ood_threshold]
        ood_df["max_confidence"] = ood_df.apply(lambda r: max(r["confidence"]), axis=1)
        # ood_df = ood_df.sort_values(by=["confidence"], ascending=False).head(img_count)
        ood_df.sort_values(by=["max_confidence"], inplace=True, ascending=False)
    else:
        ood_df.sort_values(by=[types.OoDScoreType.name()], inplace=True, ascending=ascending)

    img_count = min(60, ood_df.shape[0])
    for _, row in ood_df.head(img_count).iterrows():
        score = round(row[types.OoDScoreType.name()], 3)
        info = ImageInfo(path=row[types.RelativePathType.name()], score=score, probs=row["confidence"],
                         labels=labels, absolute_path=absolute_path_to_dataset, metadata_dir=metadata_dir)
        images_info.append(info)

    return images_info


class ShowImagesFrame(QFrame):
    def __init__(self, parent):
        super(ShowImagesFrame, self).__init__(parent)

        self.settings = PathsSettings()
        self.additional_settings = AdditionalSettings()

        self.setFrameShape(QFrame.StyledPanel)
        self.resize(100, 100)
        self.layout = QVBoxLayout()
        self.images_widget = ImagesGrid(self)
        self.__add_show_button()

        self.legend_layout = QGridLayout()
        self.layout.addLayout(self.legend_layout)

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.images_widget)
        self.layout.addWidget(self.scroll)

        self.setLayout(self.layout)

    def __add_show_button(self):
        buttons_layout = QHBoxLayout()

        self.ood_button = QPushButton("Show Worst")
        self.ood_button.clicked.connect(lambda: self.__show(False))
        buttons_layout.addWidget(self.ood_button)

        self.ood_button2 = QPushButton("Show Best")
        self.ood_button2.clicked.connect(lambda: self.__show(True))
        buttons_layout.addWidget(self.ood_button2)

        self.ood_button3 = QPushButton("Show high conf OoD")
        self.ood_button3.clicked.connect(lambda: self.__show(with_high_confidence=True))
        buttons_layout.addWidget(self.ood_button3)

        self.layout.addLayout(buttons_layout)

    def __show(self, ascending=False, with_high_confidence=False):
        self.images_widget.clear_layout()

        probabilities_file = find_pkl.get_embeddings_file(self.settings.metadata_folder)
        ood_file = find_pkl.get_ood_file(self.settings.metadata_folder)
        labels = find_pkl.get_classes_from_metadata_file(self.settings.metadata_folder)
        if not os.path.isfile(probabilities_file):
            return

        legend_df = pd.read_pickle(probabilities_file)
        legend_colors = self.__create_legend(labels)

        images_meta = get_images_to_show(ood_file, ascending, labels, self.settings.dataset_root_path,
                                         self.settings.metadata_folder, self.additional_settings.ood_threshold,
                                         with_high_confidence, legend_df, self.additional_settings.use_gt_labels)
        self.images_widget.show_images(images_meta, legend_colors)

    def analyze_settings_changed(self, settings):
        self.additional_settings = settings

    def settings_changed(self, settings):
        self.settings = settings

    def __add_label(self, label_color, label_text, row, column):
        label = QLabel(self)
        pixmap = ImagesGrid.create_label_background(color=label_color)
        label.setPixmap(pixmap.scaledToHeight(50))
        label.resize(pixmap.width(),
                     pixmap.height())
        self.legend_layout.addWidget(label, row, column)
        text = QLabel(label_text, self)
        text.setStyleSheet("font: 12pt;")
        # text.setStyleSheet("background-image: url(./{0});".format(label_img))
        self.legend_layout.addWidget(text, row + 1, column)

    def __clean_legend_layout(self):
        while self.legend_layout.count():
            child = self.legend_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def __create_legend(self, labels):
        r = lambda: random.randint(0, 255)
        colors_list = [QColor('#{:02x}{:02x}{:02x}'.format(r(), r(), r())) for random_color in range(len(labels))]
        legend_colors = []
        self.__clean_legend_layout()
        columns_count = 4
        image_row_indices = range(0, len(labels), 2)
        for idx in range(len(labels)):
            current_column_id = idx % columns_count
            current_row_id = math.floor(idx / columns_count)
            current_row_id = image_row_indices[current_row_id]
            self.__add_label(colors_list[idx], labels[idx], current_row_id, current_column_id)
            legend_colors.append(colors_list[idx])
        return legend_colors


class HistogramFrame(QFrame):
    def __init__(self, parent):
        super(HistogramFrame, self).__init__(parent)

        self.settings = PathsSettings()

        self.setFrameShape(QFrame.StyledPanel)
        self.resize(100, 100)
        self.layout = QVBoxLayout()
        self.__show_hist()
        self.setLayout(self.layout)

    def __update_hist(self):
        self._ax.cla()
        self._ax.set_xlabel("OoD Score")
        self._ax.grid(True)

        if not os.path.exists(self.settings.metadata_folder):
            return

        ood_file = find_pkl.get_ood_file(self.settings.metadata_folder)
        if not os.path.isfile(ood_file):
            return
        ood_df = pd.read_pickle(ood_file)
        ood_score = ood_df[types.OoDScoreType.name()].values
        n, bins, patches = self._ax.hist(
            ood_score, 50, density=1, facecolor="green", alpha=0.75
        )
        self._canvas.draw()

    def __show_hist(self):
        self._canvas = FigureCanvasQTAgg(Figure(figsize=(5, 3)))
        self._ax = self._canvas.figure.subplots()
        self._ax.set_xlabel("OoD Score")
        self._ax.grid(True)
        self.layout.addWidget(self._canvas)

    def analyze_settings_changed(self, settings):
        self.settings = settings
        self.__update_hist()
