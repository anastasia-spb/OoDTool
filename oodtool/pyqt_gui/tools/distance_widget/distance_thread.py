from PyQt5.QtCore import QThread, pyqtSignal
from oodtool.core.distance_wrapper.calculate_distances import DistanceCalculator


class DistanceThread(QThread, DistanceCalculator):
    _signal = pyqtSignal(int)
    _progress_signal = pyqtSignal(list)

    # constructor
    def __init__(self, embeddings_pkl, output_folder, num_neighbours):
        super(DistanceThread, self).__init__(embeddings_pkl=embeddings_pkl,
                                             output_folder=output_folder,
                                             num_neighbours=num_neighbours)

    def run(self):
        try:
            super(DistanceThread, self).get_pdist(self._progress_signal.emit)
            self._signal.emit(0)
        except Exception as error:
            print(str(error))
            self._signal.emit(1)
