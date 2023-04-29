from PyQt5.QtCore import QThread, pyqtSignal

from tool.core.model_wrappers.embedder_pipeline import EmbedderPipeline


class EmbedderPipelineThread(QThread, EmbedderPipeline):
    _signal = pyqtSignal(list)

    # constructor
    def __init__(self, metadata_file: str, data_dir: str, model_wrapper_name, use_cuda: bool, **kwargs):
        super(EmbedderPipelineThread, self).__init__(metadata_file=metadata_file, data_dir=data_dir,
                                                     model_wrapper_name=model_wrapper_name, use_cuda=use_cuda, **kwargs)

    def run(self):
        try:
            super(EmbedderPipelineThread, self).predict(self._signal.emit)
            self._signal.emit([100, 100])
        except Exception as error:
            print(str(error))
            self._signal.emit([-1, 1])
