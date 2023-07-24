from PyQt5.QtCore import QThread, pyqtSignal

from oodtool.core.data_projectors.data_projector import DataProjector


class DataProjectorThread(QThread, DataProjector):
    _signal = pyqtSignal(int)

    # constructor
    def __init__(self, method_name: str, metadata_folder, embeddings_file):
        super(DataProjectorThread, self).__init__(method_name=method_name)
        self.metadata_folder = metadata_folder
        self.embeddings_file = embeddings_file

    def run(self):
        try:
            super(DataProjectorThread, self).project(self.metadata_folder, self.embeddings_file)
            self._signal.emit(0)
        except Exception as error:
            print(str(error))
            self._signal.emit(1)
