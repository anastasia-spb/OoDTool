from PyQt5.QtCore import QThread, pyqtSignal

from tool.core.distance_wrapper.calculate_distances import DistanceCalculator


class DistanceThread(QThread, DistanceCalculator):
    _signal = pyqtSignal(int)

    # constructor
    def __init__(self, method_name, embeddings_pkl, output_folder):
        super(DistanceThread, self).__init__(method_name=method_name, embeddings_pkl=embeddings_pkl,
                                             output_folder=output_folder)

    def run(self):
        try:
            super(DistanceThread, self).get_pdist()
            self._signal.emit(0)
        except Exception as error:
            print(str(error))
            self._signal.emit(1)
