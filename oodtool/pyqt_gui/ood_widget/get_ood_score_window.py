import os
import pandas as pd
import logging
import subprocess

from typing import List, Tuple, Optional

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QMessageBox,
    QLineEdit,
    QWidget,
    QMainWindow,
    QLabel,
    QProgressBar,
    QFrame,
    QPlainTextEdit,
    QComboBox,
    QGridLayout, QCheckBox
)

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread, pyqtSignal

from oodtool.core.czebra_adapter import determine_dataset_usecase
from oodtool.core.czebra_adapter import usecase

from oodtool.core import data_types
from oodtool.core.ood_score import score_by_ood, ood_score_to_df, store_ood

from oodtool.pyqt_gui.qt_utils import find_pkl
from oodtool.pyqt_gui.ood_widget.embedder_thread import EmbedderPipelineThread
from oodtool.core.ood_score import features_selector


class JupyterThread(QThread):
    def __init__(self):
        super(JupyterThread, self).__init__()
        self.proc = None

    def run(self):
        command = ["jupyter", "notebook", "oodtool/core/ood_score/notebooks/OoDExperimental.ipynb"]
        self.proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def terminate(self):
        if self.proc is not None:
            self.proc.terminate()


class OoDThread(QThread):
    _signal = pyqtSignal(list)
    _final_signal = pyqtSignal(str)
    _log_signal = pyqtSignal(str)

    # constructor
    def __init__(self, embeddings_files: List[str], probabilities_files: List[str], metadata_dir: str,
                 metadata_df: pd.DataFrame, method: str, head_idx: Optional[int] = None):
        super(OoDThread, self).__init__()
        self.embeddings_files = embeddings_files
        self.metadata_dir = metadata_dir
        self.metadata_df = metadata_df
        self.method = method
        self.probabilities_files = probabilities_files
        self.head_idx = head_idx

    def run(self):
        try:
            result = ""
            score = score_by_ood(self.method, self.metadata_df, embeddings_files=self.embeddings_files,
                                 probabilities_files=self.probabilities_files, head_idx=self.head_idx,
                                 progress_callback=self._signal.emit, logs_callback=self._log_signal.emit)
            if score is not None:
                ood_df = ood_score_to_df(score, self.metadata_df)
                result = store_ood(ood_df, self.method, self.metadata_dir)
            self._final_signal.emit(result)
        except Exception as error:
            print(str(error))
            self._final_signal.emit("")


class OoDWidget(QWidget):
    def __init__(self, parent, metadata_dir: str, data_dir: str, head_idx: int = 0):
        super(OoDWidget, self).__init__(parent)

        self.parent = parent
        self.layout = QVBoxLayout()

        self.data_dir = data_dir
        self.jupyter_thread = None

        self.metadata_dir = metadata_dir
        try:
            metadata_file = find_pkl.get_metadata_file(self.metadata_dir)
        except FileNotFoundError:
            self.__show_error_and_exit()
            return

        if not os.path.isfile(metadata_file):
            self.__show_error_and_exit()
            return

        self.metadata_df = pd.read_pickle(metadata_file)

        if len(self.metadata_df) == 0:
            self.__show_error_and_exit()
            return

        self.ood_output = ''
        self.ood_thread = None
        self.embedder_thread = None
        self.head_idx = head_idx

        # ======== Select OoD algo =============

        algo_layout = QGridLayout()

        text = QLabel("Method: ", self)
        text.setMaximumWidth(100)
        algo_layout.addWidget(text, 0, 0)

        self.models_combobox = QComboBox()
        self.models_combobox.addItems(features_selector.OOD_METHODS)
        algo_layout.addWidget(self.models_combobox, 0, 1)

        info_button = QPushButton()
        info_button.setMaximumWidth(100)
        info_button.setIcon(QIcon("oodtool/pyqt_gui/gui_graphics/jupyter.png"))
        info_button.clicked.connect(self.__open_notebook)
        algo_layout.addWidget(info_button, 0, 2)

        # ======== Select Use Case =============

        text = QLabel("Usecase: ", self)
        text.setMaximumWidth(100)
        algo_layout.addWidget(text, 1, 0)

        self.usecase_combobox = QComboBox()
        labels = self.metadata_df[data_types.LabelsType.name()][0]
        self.usecase = determine_dataset_usecase(labels)
        if self.usecase is not None:
            self.usecase_combobox.addItems(usecase.USECASES)
            self.usecase_combobox.setCurrentText(self.usecase)

        algo_layout.addWidget(self.usecase_combobox, 1, 1)

        self.device_box = QCheckBox("Use cuda")
        self.device_box.setChecked(True)
        algo_layout.addWidget(self.device_box, 1, 2)
        self.layout.addLayout(algo_layout)

        self.apply_button = QPushButton("Score")
        self.apply_button.clicked.connect(self.__calculate_ood)
        self.layout.addWidget(self.apply_button)

        # ======== Log Window Layout ================

        self.logs_text = QPlainTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setMinimumHeight(120)
        self.layout.addWidget(self.logs_text)

        # ======== Results Layout ================
        results_container = QFrame()
        results_container.setFrameShape(QFrame.StyledPanel)
        results_container_layout = QVBoxLayout(results_container)

        self.pbar = QProgressBar(self)
        self.pbar.setValue(0)
        results_container_layout.addWidget(self.pbar)

        self.ood_file_line = QLineEdit(self.ood_output)
        self.ood_file_line.setStyleSheet("color: gray")
        self.ood_file_line.returnPressed.connect(self.__set_ood_file_back)
        results_container_layout.addWidget(self.ood_file_line)

        self.layout.addWidget(results_container)

        self.setLayout(self.layout)

    def __open_notebook(self):
        if self.jupyter_thread is None:
            self.jupyter_thread = JupyterThread()
            self.jupyter_thread.start()

    def terminate_notebook(self):
        if self.jupyter_thread is not None:
            self.jupyter_thread.terminate()
            self.jupyter_thread.exit()
            self.jupyter_thread = None


        # if selected_algo == features_selector.OOD_ENTROPY:
        #    webbrowser.open("https://arxiv.org/abs/2002.03103")
        # elif selected_algo == features_selector.OOD_KNN_DIST:
        #    webbrowser.open("https://arxiv.org/pdf/2207.03061.pdf")
        # elif selected_algo == features_selector.OOD_CONFIDENT_LEARNING:
        #    webbrowser.open("https://cleanlab.ai/blog/ood-classifier/")

    def __show_error_and_exit(self):
        if not os.path.exists(self.metadata_dir):
            text = QLabel("Directory '{0}' doesn't exists".format(self.metadata_dir), self)
        else:
            text = QLabel("No .emb.pkl files found in {0}".format(self.metadata_dir), self)
        self.layout.addWidget(text, alignment=Qt.AlignmentFlag.AlignCenter)

        exit_button = QPushButton("Close")
        exit_button.clicked.connect(self.__close)
        self.layout.addWidget(exit_button)

        self.setLayout(self.layout)

    def __close(self):
        self.parent.close()

    def __set_ood_file_back(self):
        self.ood_file_line.setText(self.ood_output)

    def __calculate_ood(self):
        self.logs_text.clear()
        self.pbar.setValue(0)
        self.apply_button.setEnabled(False)
        self.selected_ood_method = self.models_combobox.currentText()
        self.usecase = self.usecase_combobox.currentText()
        try:
            embedders_ids = features_selector.OOD_METHOD_FEATURES[self.selected_ood_method][self.usecase]
        except ValueError:
            self.apply_button.setEnabled(True)
            logging.info("Unknown usecase-ood_method combination")
            return

        device = 0 if self.device_box.isChecked() else -1
        self.embedder_thread = EmbedderPipelineThread(img_df=self.metadata_df, data_dir=self.data_dir,
                                                      output_dir=self.metadata_dir, embedders_ids=embedders_ids,
                                                      device=device)
        self.embedder_thread._progress_signal.connect(self.__update_progress_bar)
        self.embedder_thread._final_signal.connect(self.__start_ood)
        self.embedder_thread._log_signal.connect(self.log_data)
        self.embedder_thread.start()

    def __start_ood(self, embedder_files: List[Tuple[str, str]]):
        self.pbar.setValue(0)

        if len(embedder_files) == 0:
            self.log_data(f"Selected usecase \"{self.usecase}\" either is not supported for selected OoD method "
                          f"\"{self.selected_ood_method}\" or there is an error in data. "
                          f"See log file for details.")
            self.apply_button.setEnabled(True)
            return

        embeddings_files = [emb_file[0] for emb_file in embedder_files]
        probabilities_files = [emb_file[1] for emb_file in embedder_files]

        self.ood_thread = OoDThread(embeddings_files=embeddings_files,
                                    probabilities_files=probabilities_files,
                                    metadata_dir=self.metadata_dir,
                                    metadata_df=self.metadata_df,
                                    method=self.selected_ood_method,
                                    head_idx=self.head_idx)
        self.ood_thread._signal.connect(self.__update_progress_bar)
        self.ood_thread._final_signal.connect(self.signal_accept)
        self.ood_thread._log_signal.connect(self.log_data)
        self.ood_thread.start()

    def is_thread_active(self) -> bool:
        is_running = (self.ood_thread is not None) and (self.ood_thread.isRunning())
        is_running |= (self.embedder_thread is not None) and (self.embedder_thread.isRunning())
        return is_running

    def __update_progress_bar(self, progress: List[int]):
        n = progress[0]
        if n == -1:
            # Method doesn't provide progress status
            self.pbar.setFormat('Scoring in progress...')
            self.pbar.setAlignment(Qt.AlignCenter)
            self.pbar.setDisabled(True)
        total = progress[1]
        progress = int(100.0 * n / total)
        self.pbar.setValue(progress)

    def signal_accept(self, output_file):
        if os.path.isfile(output_file) and output_file.endswith(".ood.pkl"):
            self.ood_output = output_file
            self.ood_file_line.setText(self.ood_output)
        else:
            QMessageBox.warning(self, "OoDScore", "Failed")

        self.pbar.resetFormat()
        self.pbar.setEnabled(True)
        self.apply_button.setEnabled(True)

    def log_data(self, data: str):
        self.logs_text.appendPlainText(data)


class OoDScoreWindow(QMainWindow):
    def __init__(self, parent, metadata_dir: str, data_dir: str, head_idx: int = 0):
        super(OoDScoreWindow, self).__init__(parent)

        self.main_widget = OoDWidget(self, metadata_dir, data_dir, head_idx)
        self.setCentralWidget(self.main_widget)
        self.setWindowTitle("OoD")

        self.resize(600, 400)

    def closeEvent(self, event):
        if self.main_widget.is_thread_active():
            ret = QMessageBox.question(self, 'OoD is running', "Do you want to interrupt process?",
                                       QMessageBox.Yes | QMessageBox.Cancel)
            if ret == QMessageBox.Cancel:
                event.ignore()
            if ret == QMessageBox.Yes:
                self.main_widget.terminate_notebook()
                super(OoDScoreWindow, self).closeEvent(event)
        else:
            self.main_widget.terminate_notebook()
            super(OoDScoreWindow, self).closeEvent(event)
