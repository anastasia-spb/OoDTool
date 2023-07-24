import os
import pandas as pd
import time
from typing import List, Optional
from PyQt5.QtCore import QThread, pyqtSignal

from oodtool.core.towhee_adapter import TowheeAdapter


class EmbedderPipelineThread(QThread):
    _progress_signal = pyqtSignal(list)
    _final_signal = pyqtSignal(list)
    _log_signal = pyqtSignal(str)

    # constructor
    def __init__(self, img_df: pd.DataFrame, data_dir: str, output_dir: str, embedders_ids: List[str],
                 device: Optional[int] = None):
        super().__init__()
        self.embedder_wrapper = TowheeAdapter(img_df, data_dir, output_dir)
        self.embedders_ids = embedders_ids
        self.device = device

    def get_files_if_exist(self, emb_id: str):
        output_file = self.embedder_wrapper.get_filename(emb_id)
        if os.path.isfile(output_file):
            self._log_signal.emit(f"File for {emb_id} exists.")
            output_probabilities_file = self.embedder_wrapper.get_probabilities_filename(emb_id)
            if not os.path.isfile(output_probabilities_file):
                output_probabilities_file = None
            return output_file, output_probabilities_file
        return None, None

    def run(self):
        try:
            results = []
            for emb_id in self.embedders_ids:
                output_file, output_probabilities_file = self.get_files_if_exist(emb_id)
                if output_file is not None:
                    results.append((output_file, output_probabilities_file))
                    continue

                self._log_signal.emit(f"Starting embedder {emb_id}...")
                start_time = time.perf_counter()
                output_file, output_probabilities_file = self.embedder_wrapper.predict(
                    model_id=emb_id, progress_callback=self._progress_signal.emit, device=self.device)
                results.append((output_file, output_probabilities_file))
                end_time = time.perf_counter()
                self._log_signal.emit(f"Embedder {emb_id} finished in {end_time - start_time:0.4f} seconds")

            self._final_signal.emit(results)
        except Exception as error:
            print(str(error))
            self._final_signal.emit([])
