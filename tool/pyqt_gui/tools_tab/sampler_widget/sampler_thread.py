from PyQt5.QtCore import QThread, pyqtSignal

from tool.core.density_based_sampler.sampler import DensityBasedSampler


class SamplerThread(QThread, DensityBasedSampler):
    _signal = pyqtSignal(int)

    # constructor
    def __init__(self, embeddings_file, ood_file, n_samples=300):
        super(SamplerThread, self).__init__(embeddings_file=embeddings_file, ood_score_file=ood_file)
        self.n_samples = n_samples
        self.result = ''

    def run(self):
        try:
            self.result = super(SamplerThread, self).fit(n_samples=self.n_samples)
            self._signal.emit(0)
        except Exception as error:
            print(str(error))
            self._signal.emit(1)
