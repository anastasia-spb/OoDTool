from typing import Optional, List

from PyQt5.QtCore import QThread, pyqtSignal

from tool.core.classifier_wrappers.classifier_pipeline import ClassifierPipeline


class ClassifierThread(QThread, ClassifierPipeline):
    _signal = pyqtSignal(list)

    # constructor
    def __init__(self, classifier_tag, embeddings_files: List[str], output_dir: str, use_gt_for_training: bool,
                 probabilities_file: Optional[str], kwargs: List[dict]):
        super(ClassifierThread, self).__init__(classifier_tag=classifier_tag)
        self.embeddings_files = embeddings_files
        self.output_dir = output_dir
        self.use_gt_for_training = use_gt_for_training
        self.probabilities_file = probabilities_file
        self.kwargs = kwargs

    def run(self):
        try:
            results = super(ClassifierThread, self).train_and_classify(
                embeddings_files=self.embeddings_files,
                output_dir=self.output_dir,
                use_gt_for_training=self.use_gt_for_training,
                probabilities_file=self.probabilities_file,
                kwargs=self.kwargs)
            self._signal.emit(results)
        except Exception as error:
            print(str(error))
            self._signal.emit([])
