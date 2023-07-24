from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QFrame,
    QLineEdit,
    QLabel, QComboBox, QMessageBox, QMainWindow, QHBoxLayout, QProgressBar
)
from PyQt5.QtCore import Qt
from oodtool.pyqt_gui.tools.distance_widget.distance_thread import DistanceThread
from oodtool.pyqt_gui.qt_utils import find_pkl

from typing import List


class DistanceFrame(QFrame):

    def __init__(self, parent, metadata_dir: str):
        super(DistanceFrame, self).__init__(parent)

        spacing_between_layouts = 30

        self.output_file = ''

        self.metadata_dir = metadata_dir
        self.embeddings_files = find_pkl.get_embeddings_files(self.metadata_dir)
        if len(self.embeddings_files) == 0:
            self.selected_embeddings = None
        else:
            self.selected_embeddings = self.embeddings_files[0]

        self.setFrameShape(QFrame.StyledPanel)
        self.layout = QVBoxLayout()

        self.num_neighbours = 10

        self.__embeddings_selection_layout(self.embeddings_files)
        self.layout.addSpacing(spacing_between_layouts)
        self.__neighbours_to_store(self.num_neighbours)
        self.layout.addSpacing(spacing_between_layouts)
        self.__add_generate_metadata_button()
        self.layout.addSpacing(spacing_between_layouts)

        self.pbar = QProgressBar(self)
        self.pbar.setValue(0)
        self.layout.addWidget(self.pbar)

        self.__add_output_line()

        self.setLayout(self.layout)

    def __neighbours_to_store(self, num):
        num_images_layout = QHBoxLayout()

        num_images_text = QLabel("Max number of neighbours to store/show: ", self)
        num_images_layout.addWidget(num_images_text, alignment=Qt.AlignmentFlag.AlignLeft)

        self.num_images_to_show_line = QLineEdit()
        self.num_images_to_show_line.setText(str(num))
        num_images_layout.addWidget(self.num_images_to_show_line, alignment=Qt.AlignmentFlag.AlignLeft)

        self.layout.addLayout(num_images_layout)

    def __set_embeddings_file_back(self):
        self.output_file_line.setText(self.output_file)

    def __add_output_line(self):
        self.output_file_line = QLineEdit(self.output_file)
        self.output_file_line.setStyleSheet("color: gray")
        self.output_file_line.returnPressed.connect(self.__set_embeddings_file_back)
        self.layout.addWidget(self.output_file_line)

    def __add_generate_metadata_button(self):
        self.generate_button = QPushButton("Find")
        self.generate_button.setEnabled(True)
        self.generate_button.clicked.connect(self.__fit_embeddings)
        self.layout.addWidget(self.generate_button)

    def __fit_embeddings(self):
        if self.selected_embeddings is not None:
            self.generate_button.setEnabled(False)

            t = self.num_images_to_show_line.text()
            try:
                num_images_to_show = min(max(0, int(t)), 100)
            except ValueError:
                num_images_to_show = self.settings.num_images_to_show

            self.projector_thread = DistanceThread(embeddings_pkl=self.selected_embeddings,
                                                   output_folder=self.metadata_dir,
                                                   num_neighbours=num_images_to_show)

            self.projector_thread._signal.connect(self.signal_accept)
            self.projector_thread._progress_signal.connect(self.__update_progress_bar)
            self.projector_thread.start()
        else:
            QMessageBox.warning(self, "No .emb.pkl files found.",
                                "Please, launch 'OoD Score' tool for embeddings generation or "
                                "open another session folder.")

    def __update_progress_bar(self, progress: List[int]):
        n = progress[0]
        total = progress[1]
        progress = int(100.0 * n / total)
        self.pbar.setValue(progress)

    def signal_accept(self, msg):
        if msg == 0:
            self.output_file_line.setText(self.projector_thread.get_output_file())
        else:
            QMessageBox.warning(self, "DistanceCalculation", "Failed")

        self.generate_button.setEnabled(True)

    def __embeddings_selection_layout(self, embeddings):
        self.embeddings_combobox = QComboBox()
        self.embeddings_combobox.currentTextChanged.connect(self.__on_embedding_file_change)
        self.embeddings_combobox.addItems(embeddings)
        self.layout.addWidget(self.embeddings_combobox)

    def __on_embedding_file_change(self, value):
        self.selected_embeddings = value


class NeigboursWindow(QMainWindow):
    def __init__(self, parent, metadata_dir: str):
        super(NeigboursWindow, self).__init__(parent)

        self.main_widget = DistanceFrame(self, metadata_dir)
        self.setCentralWidget(self.main_widget)
        self.setWindowTitle("Neighbours Search")

        # self.resize(600, 400)
