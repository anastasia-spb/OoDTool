from dataclasses import dataclass


@dataclass
class ImageInfo:
    relative_path: str
    ood_score: float
    probabilities: list
    labels: list
    absolute_path: str
    metadata_dir: str

    def __init__(self, path: str, score: float, probs: list, labels: list, absolute_path: str, metadata_dir: str):
        self.relative_path = path
        self.ood_score = score
        self.probabilities = probs
        self.labels = labels
        self.absolute_path = absolute_path
        self.metadata_dir = metadata_dir
