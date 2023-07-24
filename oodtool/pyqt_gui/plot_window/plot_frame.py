import numpy as np
import random
import seaborn as sns

from typing import Dict, Optional, List

import logging

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QFrame,
)

from PyQt5.QtGui import QColor

from oodtool.pyqt_gui.plot_window.plot_settings import AdditionalSettings
from oodtool.pyqt_gui.data_loader.loader import DataLoader

from oodtool.pyqt_gui.ood_images_tab.filters_frame import FilterSettings


class ScatterProperties:
    plot_key: str
    indices: List[int]
    marker_type: str
    edgecolor: str
    cmap: str

    def __init__(self, plot_key: str, indices: List[int], marker_type: str, edgecolor: str, cmap: str):
        super(ScatterProperties, self).__init__()

        self.plot_key = plot_key
        self.indices = indices
        self.marker_type = marker_type
        self.edgecolor = edgecolor
        self.cmap = cmap


class PlotFrame(QFrame):
    def __init__(self, parent, data_loader: DataLoader, legend: Optional[Dict[str, QColor]] = None,
                 filter_settings: Optional[FilterSettings] = None, head_idx: int = 0):
        super(PlotFrame, self).__init__(parent)

        self.parent = parent

        self.analyze_settings = AdditionalSettings()

        self.setFrameShape(QFrame.StyledPanel)
        self.resize(100, 100)
        self.layout = QVBoxLayout()
        self.__show_hist()

        self.setLayout(self.layout)

        self.__init(data_loader, legend, filter_settings, head_idx)

    def reload_plot(self, data_loader: DataLoader, legend: Optional[Dict[str, QColor]] = None,
                    filter_settings: Optional[FilterSettings] = None, head_idx: int = 0):
        self.__init(data_loader, legend, filter_settings, head_idx)

    def __init(self, data_loader: DataLoader, legend: Optional[Dict[str, QColor]] = None,
               filter_settings: Optional[FilterSettings] = None, head_idx: int = 0):
        self.data_loader = data_loader
        self.legend = legend
        self.filter_settings = filter_settings

        self.scatter_dict = dict()
        self.scatter_plots_keys = ["".join(('_child', str(index))) for index in range(200)]

        if self.loaded_data_valid():
            self.__prepare_scatter_dict(head_idx)
            self.__update_plot(self.analyze_settings.highlight_test_samples)

    def loaded_data_valid(self):
        return (self.data_loader is not None) and self.data_loader.status_loaded and \
            (self.data_loader.get_projected_data_points() is not None)

    def __prepare_scatter_dict(self, head_idx: int = 0):
        # Each value in dictionary contains list of original indices and scatter properties
        self.scatter_dict = dict()
        self.plot_key_to_scatter = dict()
        scatter_plot_idx = 0

        correctly_predicted_indices = self.data_loader.get_indices_of_data_with_correct_prediction(head_idx)
        if (self.filter_settings is not None) and (not self.filter_settings.show_test):
            test_indices = []
            train_indices = self.data_loader.get_train_indices()
        elif (self.filter_settings is not None) and (not self.filter_settings.show_train):
            test_indices = self.data_loader.get_test_indices()
            train_indices = []
        else:
            test_indices = self.data_loader.get_test_indices()
            train_indices = self.data_loader.get_train_indices()

        labels = self.data_loader.get_labels()
        if labels is None:
            return

        for label in labels:
            if self.filter_settings is not None:
                if label not in self.filter_settings.selected_labels:
                    continue

            self.scatter_dict[label] = dict()

            label_indices = self.data_loader.get_indices([label])

            # Generate colormap for each label
            if self.legend is not None:
                scatter_q_color = self.legend[label]
                name = scatter_q_color.name()
                scatter_color_map = sns.color_palette(f"light:{name}", as_cmap=True)
            else:
                name = hex(random.randrange(0, 2 ** 24))
                scatter_color_map = sns.color_palette(f"light:{name}", as_cmap=True)

            # Find indices with required properties
            label_train_indices = list(set(label_indices) & set(train_indices))
            label_test_indices = list(set(label_indices) & set(test_indices))

            if correctly_predicted_indices is not None:
                label_train_correctly_predicted_indices = list(
                    set(label_train_indices) & set(correctly_predicted_indices))
                label_test_correctly_predicted_indices = list(
                    set(label_test_indices) & set(correctly_predicted_indices))
                label_train_miss_indices = list(set(label_train_indices) - set(correctly_predicted_indices))
                label_test_miss_indices = list(set(label_test_indices) - set(correctly_predicted_indices))
            else:
                label_train_correctly_predicted_indices = label_train_indices
                label_test_correctly_predicted_indices = label_test_indices
                label_train_miss_indices = []
                label_test_miss_indices = []

            plot_key = "".join(('_child', str(scatter_plot_idx)))
            self.scatter_dict[label]["train_correct"] = ScatterProperties(
                plot_key=plot_key, indices=label_train_correctly_predicted_indices, marker_type='o', edgecolor=None,
                cmap=scatter_color_map)
            self.plot_key_to_scatter[plot_key] = (label, "train_correct")
            scatter_plot_idx += 1

            plot_key = "".join(('_child', str(scatter_plot_idx)))
            self.scatter_dict[label]["train_miss"] = ScatterProperties(
                plot_key=plot_key, indices=label_train_miss_indices, marker_type='X', edgecolor=None,
                cmap=scatter_color_map)
            self.plot_key_to_scatter[plot_key] = (label, "train_miss")
            scatter_plot_idx += 1

            plot_key = "".join(('_child', str(scatter_plot_idx)))
            self.scatter_dict[label]["test_correct"] = ScatterProperties(
                plot_key=plot_key, indices=label_test_correctly_predicted_indices, marker_type='o', edgecolor='red',
                cmap=scatter_color_map)
            self.plot_key_to_scatter[plot_key] = (label, "test_correct")
            scatter_plot_idx += 1

            plot_key = "".join(('_child', str(scatter_plot_idx)))
            self.scatter_dict[label]["test_miss"] = ScatterProperties(
                plot_key=plot_key, indices=label_test_miss_indices, marker_type='X', edgecolor='red',
                cmap=scatter_color_map)
            self.plot_key_to_scatter[plot_key] = (label, "test_miss")
            scatter_plot_idx += 1

    def __update_plot(self, highlight_test):
        self._ax.cla()  # Clear axis
        self._ax.grid(True)

        for key1, key2 in self.plot_key_to_scatter.values():
            properties = self.scatter_dict[key1][key2]
            x = np.take(self.data_loader.get_projected_data_points(), properties.indices, axis=0)
            score_values = self.data_loader.get_ood_score()
            if score_values is not None:
                if self.analyze_settings.ood_threshold > 0.0:
                    ood_score_cmap = [0.1 if score < self.analyze_settings.ood_threshold else 0.9 for score in
                                      score_values]
                    ood_score = np.take(ood_score_cmap, properties.indices, axis=0)
                else:
                    ood_score = np.take(score_values, properties.indices, axis=0)
                c_values = [properties.cmap(value) for value in ood_score]
            else:
                c_values = [1.0] * x.shape[0]

            if highlight_test:
                edgecolors = properties.edgecolor
            else:
                edgecolors = None
            self._ax.scatter(x[:, 0], x[:, 1], c=c_values, picker=True, pickradius=3,
                             alpha=0.7, marker=properties.marker_type, edgecolors=edgecolors)

        self._canvas.draw()

    def __onpick(self, event):
        ind = event.ind[0]
        key1, key2 = self.plot_key_to_scatter[event.artist.get_label()]
        properties = self.scatter_dict[key1][key2]
        global_index = properties.indices[ind]

        info = self.data_loader.get_images_info_at(global_index)
        if info is not None:
            self.parent.image_path_signal.emit(info)

    def __show_hist(self):
        self._canvas = FigureCanvasQTAgg(Figure(figsize=(5, 3)))
        self._canvas.mpl_connect('pick_event', self.__onpick)
        self._ax = self._canvas.figure.subplots()
        self._ax.grid(True)

        toolbar = NavigationToolbar(self._canvas, self)
        self.layout.addWidget(toolbar)
        self.layout.addWidget(self._canvas)

    def analyze_inputs_changed(self, settings):
        self.analyze_settings = settings
        if self.loaded_data_valid():
            self.__update_plot(self.analyze_settings.highlight_test_samples)
        else:
            self._ax.cla()  # Clear axis
            logging.info("Plot data invalid.")
