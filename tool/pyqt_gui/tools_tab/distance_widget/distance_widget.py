from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QFrame,
    QLineEdit,
    QLabel, QComboBox, QMessageBox
)

from tool.pyqt_gui.tools_tab.distance_widget.distance_thread import DistanceThread
from tool.pyqt_gui.paths_settings import PathsSettings
from tool.pyqt_gui.qt_utils import find_pkl


class DistanceFrame(QFrame):

    def __init__(self, parent):
        super(DistanceFrame, self).__init__(parent)

        spacing_between_layouts = 30
        self.settings = PathsSettings()

        self.output_file = ''
        self.selected_method = DistanceThread.method_name

        self.setFrameShape(QFrame.StyledPanel)
        self.setMaximumHeight(200)
        self.resize(100, 100)
        self.layout = QVBoxLayout()

        text = QLabel("Distance", self)
        font = text.font()
        font.setBold(True)
        text.setFont(font)
        self.layout.addWidget(text)

        self.__embeddings_selection_layout()
        self.layout.addSpacing(spacing_between_layouts)
        self.__add_generate_metadata_button()
        self.layout.addSpacing(spacing_between_layouts)

        self.__add_output_line()

        self.setLayout(self.layout)

    def ood_settings_changed(self, settings):
        self.settings = settings

    def __set_embeddings_file_back(self):
        self.output_file_line.setText(self.output_file)

    def __add_output_line(self):
        self.output_file_line = QLineEdit(self.output_file)
        self.output_file_line.setStyleSheet("color: gray")
        self.output_file_line.returnPressed.connect(self.__set_embeddings_file_back)
        self.layout.addWidget(self.output_file_line)

    def __add_generate_metadata_button(self):
        self.generate_button = QPushButton("Calculate")
        self.generate_button.setEnabled(True)
        self.generate_button.clicked.connect(self.__fit_embeddings)
        self.layout.addWidget(self.generate_button)

    def __fit_embeddings(self):
        self.generate_button.setEnabled(False)

        self.projector_thread = DistanceThread(method_name=self.selected_method,
                                               embeddings_pkl=find_pkl.get_embeddings_file(self.settings.metadata_folder),
                                               output_folder=self.settings.metadata_folder)

        self.projector_thread._signal.connect(self.signal_accept)
        self.projector_thread.start()

    def signal_accept(self, msg):
        if msg == 0:
            self.output_file_line.setText(self.projector_thread.get_output_file())
        else:
            QMessageBox.warning(self, "DistanceCalculation", "Failed")

        self.generate_button.setEnabled(True)

    def __embeddings_selection_layout(self):
        self.datasets_combobox = QComboBox()
        self.datasets_combobox.textActivated.connect(self.__on_datasets_combobox_values_change)
        self.datasets_combobox.addItems(DistanceThread.methods)
        self.layout.addWidget(self.datasets_combobox)

    def __on_datasets_combobox_values_change(self, value):
        self.selected_method = value
