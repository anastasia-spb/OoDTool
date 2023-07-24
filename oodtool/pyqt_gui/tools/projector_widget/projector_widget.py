from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QFrame,
    QLineEdit,
    QComboBox, QMessageBox, QMainWindow
)

from oodtool.pyqt_gui.tools.projector_widget.projector_thread import DataProjectorThread
from oodtool.pyqt_gui.qt_utils import find_pkl


class ProjectorFrame(QFrame):

    def __init__(self, parent, metadata_dir: str):
        super(ProjectorFrame, self).__init__(parent)

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

        self.projector_method = DataProjectorThread.method_name

        self.__embeddings_selection_layout(self.embeddings_files)
        self.layout.addSpacing(spacing_between_layouts)
        self.__select_projector_method()
        self.layout.addSpacing(spacing_between_layouts)
        self.__add_generate_metadata_button()
        self.layout.addSpacing(spacing_between_layouts)

        self.__add_output_line()

        self.setLayout(self.layout)

    def __select_projector_method(self):
        self.methods_combobox = QComboBox()
        self.methods_combobox.currentTextChanged.connect(self.__on_method_change)
        self.methods_combobox.addItems(DataProjectorThread.methods)
        self.layout.addWidget(self.methods_combobox)

    def __on_method_change(self, value):
        self.projector_method = value

    def __set_embeddings_file_back(self):
        self.output_file_line.setText(self.output_file)

    def __add_output_line(self):
        self.output_file_line = QLineEdit(self.output_file)
        self.output_file_line.setStyleSheet("color: gray")
        self.output_file_line.returnPressed.connect(self.__set_embeddings_file_back)
        self.layout.addWidget(self.output_file_line)

    def __add_generate_metadata_button(self):
        self.generate_button = QPushButton("Project")
        self.generate_button.setEnabled(True)
        self.generate_button.clicked.connect(self.__fit_embeddings)
        self.layout.addWidget(self.generate_button)

    def __fit_embeddings(self):
        if self.selected_embeddings is not None:
            self.generate_button.setEnabled(False)

            self.projector_thread = DataProjectorThread(method_name=self.projector_method,
                                                        metadata_folder=self.metadata_dir,
                                                        embeddings_file=self.selected_embeddings)

            self.projector_thread._signal.connect(self.signal_accept)
            self.projector_thread.start()
        else:
            QMessageBox.warning(self, "No .emb.pkl files found.",
                                "Please, launch 'OoD Score' tool for embeddings generation or "
                                "open another session folder.")

    def signal_accept(self, msg):
        if msg == 0:
            self.output_file_line.setText(self.projector_thread.get_output_file())
        else:
            QMessageBox.warning(self, "Data Projector", "Failed")

        self.generate_button.setEnabled(True)

    def __embeddings_selection_layout(self, embeddings):
        self.embeddings_combobox = QComboBox()
        self.embeddings_combobox.currentTextChanged.connect(self.__on_embedding_file_change)
        self.embeddings_combobox.addItems(embeddings)
        self.layout.addWidget(self.embeddings_combobox)

    def __on_embedding_file_change(self, value):
        self.selected_embeddings = value


class ProjectorWindow(QMainWindow):
    def __init__(self, parent, metadata_dir: str):
        super(ProjectorWindow, self).__init__(parent)

        self.main_widget = ProjectorFrame(self, metadata_dir)
        self.setCentralWidget(self.main_widget)
        self.setWindowTitle("Project data")

        # self.resize(600, 400)
