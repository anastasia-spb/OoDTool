import ast

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLineEdit,
    QComboBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QTextEdit,
    QCheckBox
)

from PyQt5.QtCore import Qt

from tool.pyqt_gui.tools_tab.embedder_widget.embedder_thread import EmbedderPipelineThread
from tool.pyqt_gui.qt_utils import find_pkl
from tool.pyqt_gui.paths_settings import PathsSettings


class EmbedderFrame(QFrame):
    def __init__(self, parent):
        super(EmbedderFrame, self).__init__(parent)

        self.settings = PathsSettings()
        self.empty_text = '...'
        self.embeddings_pkl_file = self.empty_text
        self.use_cuda = True

        self.setFrameShape(QFrame.StyledPanel)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.resize(100, 100)
        self.layout = QVBoxLayout()

        text = QLabel("Embedder", self)
        font = text.font()
        font.setBold(True)
        text.setFont(font)
        self.layout.addWidget(text)

        self.__kwargs_layout()
        self.__select_model_layout()
        self.__add_evaluate_button()

        self.use_cuda_box = QCheckBox("Use cuda")
        self.use_cuda_box.setChecked(self.use_cuda)
        self.use_cuda_box.stateChanged.connect(lambda: self.__check_box_state_changed())
        self.layout.addWidget(self.use_cuda_box)

        self.requires_grad = False
        self.requires_grad_box = QCheckBox("Requires grad")
        self.requires_grad_box.setChecked(self.requires_grad)
        self.requires_grad_box.stateChanged.connect(lambda: self.__grad_check_box_state_changed())
        self.layout.addWidget(self.requires_grad_box)

        self.layout.addStretch()
        self.setLayout(self.layout)

    def get_embeddings_pkl_file(self):
        if self.embeddings_pkl_file == self.empty_text:
            return None
        return self.embeddings_pkl_file

    def ood_settings_changed(self, settings):
        self.settings = settings

    def __check_box_state_changed(self):
        self.use_cuda = self.use_cuda_box.isChecked()

    def __grad_check_box_state_changed(self):
        self.requires_grad = self.requires_grad_box.isChecked()

    def __kwargs_layout(self):
        self.kwargs_hint_line = QLabel('', self)
        self.kwargs_hint_line.setWordWrap(True)
        self.layout.addWidget(self.kwargs_hint_line)

        self.kwargs_line = QTextEdit()
        self.kwargs_line.setMaximumHeight(200)
        self.layout.addWidget(self.kwargs_line)

    def __select_model_layout(self):
        model_layout = QHBoxLayout()
        models_combobox = QComboBox()
        models_combobox.currentTextChanged.connect(self.__on_model_type_change)
        models_combobox.addItems(EmbedderPipelineThread.get_supported_embedders())
        model_layout.addWidget(models_combobox)
        self.layout.insertLayout(1, model_layout)

    def __on_model_type_change(self, value):
        self.selected_model = value
        self.kwargs_hint_line.setText(EmbedderPipelineThread.get_model_text(self.selected_model))
        self.kwargs_line.setText(EmbedderPipelineThread.get_input_hint(self.selected_model))

    def __add_evaluate_button(self):
        self.calculate_embeddings_button = QPushButton("Calculate embeddings")
        self.calculate_embeddings_button.clicked.connect(self.__calculate_embeddings)
        self.layout.addWidget(self.calculate_embeddings_button)

        self.pbar = QProgressBar(self)
        self.pbar.setValue(0)
        self.layout.addWidget(self.pbar)

        self.output_file = QLineEdit(self.embeddings_pkl_file)
        self.output_file.setStyleSheet("color: gray")
        self.output_file.returnPressed.connect(self.__set_embeddings_file_back)
        self.layout.addWidget(self.output_file)

    def __set_embeddings_file_back(self):
        self.output_file.setText(self.embeddings_pkl_file)

    def __calculate_embeddings(self):
        self.calculate_embeddings_button.setEnabled(False)
        kwargs = dict()
        try:
            kwargs = ast.literal_eval(self.kwargs_line.toPlainText())
        except SyntaxError:
            QMessageBox.critical(self, "", "Parameters couldn't be parsed")

        self.embeddings_calc_thread = EmbedderPipelineThread(
            find_pkl.get_metadata_file(self.settings.metadata_folder),
            self.settings.dataset_root_path,
            self.selected_model, self.use_cuda, self.requires_grad, **kwargs)

        self.embeddings_calc_thread._signal.connect(self.signal_accept)
        self.embeddings_calc_thread.start()

    def __store_embeddings(self):
        self.embeddings_pkl_file = self.embeddings_calc_thread.get_embeddings_pkl_file()
        self.output_file.setText(self.embeddings_pkl_file)

    def signal_accept(self, msg):
        n = msg[0]
        total = msg[1]
        progress = int(100.0 * n / total)
        if progress < 0:
            QMessageBox.critical(self, "", "They were errors in Embedder. Embeddings file is not generated.")
        else:
            self.pbar.setValue(progress)
            if progress == 100:
                self.__store_embeddings()
        self.calculate_embeddings_button.setEnabled(True)
