from dataclasses import dataclass
from typing import Optional, List
from oodtool.core.data_types import types


@dataclass
class ImageInfo:
    relative_path: str
    ood_score: float
    # for each head in multihead case
    confidence: List[float]
    labels: list
    dataset_root_dir: str
    metadata_dir: str
    gt_label: str
    # for each head in multihead case
    predicted_label: List[str]

    def __init__(self,
                 relative_path: str,
                 dataset_root_dir: str,
                 metadata_dir: str,
                 ood_score: Optional[float] = None,
                 confidence: Optional[List[float]] = None,
                 labels: Optional[List[str]] = None,
                 gt_label: Optional[str] = None,
                 predicted_label: List[str] = None):
        self.relative_path = relative_path
        self.ood_score = ood_score
        self.confidence = confidence
        self.labels = labels
        self.dataset_root_dir = dataset_root_dir
        self.metadata_dir = metadata_dir
        self.gt_label = gt_label
        self.predicted_label = predicted_label

    def to_dict(self):
        return {
            types.RelativePathType.name(): self.relative_path,
            "dataset_root_dir": self.dataset_root_dir,
            "metadata_dir": self.metadata_dir,
            types.OoDScoreType.name(): self.ood_score,
            types.ClassProbabilitiesType.name(): self.confidence,
            types.PredictedLabelsType.name(): self.predicted_label,
            types.LabelsType.name(): self.labels,
            types.LabelType.name(): self.gt_label
        }
