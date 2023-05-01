import os
import numpy as np
from tool.core.data_types import types
import matplotlib as mpl

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QFrame,
    QCheckBox
)

from PyQt5.QtCore import pyqtSignal

from tool.pyqt_gui.plot_tab.visualization_preprocessing.data_reader import DataReader

from tool.pyqt_gui.paths_settings import PathsSettings
from tool.pyqt_gui.analyze_settings import AdditionalSettings

from tool.pyqt_gui.qt_utils import find_pkl
from tool.pyqt_gui.qt_utils.qt_types import ImageInfo


class PlotLegendFrame(QFrame):
    plot_legend_signal = pyqtSignal(bool)

    def __init__(self, parent):
        super(PlotLegendFrame, self).__init__(parent)

        self.layout = QVBoxLayout()

        self.highlight_box = QCheckBox("Highlight test samples")
        self.highlight_box.setChecked(False)
        self.highlight_box.stateChanged.connect(lambda: self.__check_box_state_changed())
        self.layout.addWidget(self.highlight_box)
        self.setLayout(self.layout)

    def __check_box_state_changed(self):
        highlight = self.highlight_box.isChecked()
        self.plot_legend_signal.emit(highlight)


class PlotFrame(QFrame):
    image_path_signal = pyqtSignal(ImageInfo)

    def __init__(self, parent):
        super(PlotFrame, self).__init__(parent)

        self.settings = PathsSettings()
        self.analyze_settings = AdditionalSettings()
        self.data_reader = None
        self.highlight_test_samples = False

        self.scatter_dict = dict()
        self.scatter_plots_keys = ["".join(('_child', str(index))) for index in range(80)]

        self.setFrameShape(QFrame.StyledPanel)
        self.resize(100, 100)
        self.layout = QVBoxLayout()
        self.__show_hist()
        self.__update_plot(self.highlight_test_samples)
        self.setLayout(self.layout)

    def highlight_test_samples_req_changed(self, value):
        self.highlight_test_samples = value
        self.__update_plot(self.highlight_test_samples)

    def __split_to_train_test(self, indices, test_indices, scatter_idx, current_cmap, marker_type, edgecolor):
        idxs_correct_test = list(set(indices) & set(test_indices))
        idxs_correct_train = list(set(indices) - set(test_indices))
        if len(idxs_correct_test) > 0:
            self.scatter_dict[self.scatter_plots_keys[scatter_idx]] = (idxs_correct_test, marker_type, current_cmap,
                                                                       'red')
            scatter_idx += 1
        if len(idxs_correct_train) > 0:
            self.scatter_dict[self.scatter_plots_keys[scatter_idx]] = (idxs_correct_train, marker_type, current_cmap,
                                                                       edgecolor)
            scatter_idx += 1
        return scatter_idx

    def __update_plot(self, highlight_test):
        self._ax.cla()  # Clear axis
        self._ax.grid(True)
        if not os.path.exists(self.settings.metadata_folder):
            return
        emb_2d_file = find_pkl.get_2d_emb_file(self.settings.metadata_folder)
        ood_score_file = find_pkl.get_ood_file(self.settings.metadata_folder)
        if not os.path.isfile(emb_2d_file):
            return
        self.data_reader = DataReader(self.settings.dataset_root_path, emb_2d_file,
                                      ood_score_file, self.analyze_settings.use_gt_labels,
                                      self.analyze_settings.ood_threshold)
        labels = self.data_reader.get_label()
        cmaps = ['Greens', 'Reds', 'Greys', 'Purples', 'Blues', 'Oranges',
                 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        self.scatter_dict = dict()
        scatter_idx = 0
        cmap_idx = 0
        for i in range(len(labels)):
            # edgecolor = mpl.cm.get_cmap(cmaps[cmap_idx])(1.0)
            edgecolor = None
            label_indices = self.data_reader.get_indices_of_data_with_label(i)
            correctly_predicted_indices = self.data_reader.get_indices_of_data_with_correct_prediction()
            test_indices = None
            if highlight_test:
                test_indices = self.data_reader.get_indices_of_test_samples()
            if correctly_predicted_indices is not None:
                idxs_correct = list(set(correctly_predicted_indices) & set(label_indices))
                if len(idxs_correct) > 0:
                    if test_indices is None:
                        self.scatter_dict[self.scatter_plots_keys[scatter_idx]] = (idxs_correct, 'o', cmaps[cmap_idx],
                                                                                   edgecolor)
                        scatter_idx += 1
                    else:
                        scatter_idx = self.__split_to_train_test(idxs_correct, test_indices, scatter_idx,
                                                                 cmaps[cmap_idx], marker_type='o', edgecolor=edgecolor)

                idxs_wrong = list(set(label_indices) - set(correctly_predicted_indices))
                if len(idxs_wrong) > 0:
                    if test_indices is None:
                        self.scatter_dict[self.scatter_plots_keys[scatter_idx]] = (idxs_wrong, 'X', cmaps[cmap_idx],
                                                                                   edgecolor)
                        scatter_idx += 1
                    else:
                        scatter_idx = self.__split_to_train_test(idxs_wrong, test_indices, scatter_idx,
                                                                 cmaps[cmap_idx], marker_type='X', edgecolor=edgecolor)
            else:
                if test_indices is None:
                    self.scatter_dict[self.scatter_plots_keys[scatter_idx]] = (label_indices, 'o', cmaps[cmap_idx],
                                                                               edgecolor)
                    scatter_idx += 1
                else:
                    scatter_idx = self.__split_to_train_test(label_indices, test_indices, scatter_idx,
                                                             cmaps[cmap_idx], marker_type='o', edgecolor=edgecolor)
            cmap_idx += 1
            if cmap_idx >= len(cmaps):
                cmap_idx = 0

        for value in self.scatter_dict.values():
            x = np.take(self.data_reader.embeddings, value[0], axis=0)
            if self.data_reader.ood_score is not None:
                if self.analyze_settings.ood_threshold > 0.0:
                    ood_score = np.take(self.data_reader.ood_score_cmap, value[0], axis=0)
                else:
                    ood_score = np.take(self.data_reader.ood_score, value[0], axis=0)
                cmap = mpl.colormaps[value[2]]
                c_values = [cmap(value) for value in ood_score]
            else:
                c_values = [1.0] * x.shape[0]
            self._ax.scatter(x[:, 0], x[:, 1], c=c_values, picker=True, pickradius=3,
                             alpha=0.7, marker=value[1], edgecolors=value[3])

        self._canvas.draw()

    def __onpick(self, event):
        ind = event.ind[0]
        indices = self.scatter_dict[event.artist.get_label()][0]
        paths = self.data_reader.data_df[types.RelativePathType.name()].values[indices]

        # print('onpick3 scatter:', ind)
        if self.data_reader.ood_score is None:
            ood_score = -1.0
        else:
            ood_score = np.take(self.data_reader.ood_score, indices, axis=0)[ind]

        if self.data_reader.probabilities is None:
            prob = [-1.0]
        else:
            prob = np.take(self.data_reader.probabilities, indices, axis=0)[ind]

        info = ImageInfo(paths[ind], ood_score, prob, self.data_reader.get_label(),
                         self.settings.dataset_root_path, self.settings.metadata_folder)
        self.image_path_signal.emit(info)

    def __show_hist(self):
        self._canvas = FigureCanvasQTAgg(Figure(figsize=(5, 3)))
        self._canvas.mpl_connect('pick_event', self.__onpick)
        self._ax = self._canvas.figure.subplots()
        self._ax.grid(True)

        toolbar = NavigationToolbar(self._canvas, self)
        self.layout.addWidget(toolbar)
        self.layout.addWidget(self._canvas)

    def inputs_changed(self, settings):
        self.settings = settings
        self.__update_plot(self.highlight_test_samples)

    def analyze_inputs_changed(self, settings):
        self.analyze_settings = settings
        self.__update_plot(self.highlight_test_samples)
