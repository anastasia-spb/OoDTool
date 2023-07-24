import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class RelativePathType:
    value: str

    @classmethod
    def name(cls):
        return 'relative_path'


@dataclass
class NeighboursType:
    value: list

    @classmethod
    def name(cls):
        return 'neighbours_paths'


@dataclass
class EmbeddingsType:
    value: np.array

    @classmethod
    def name(cls):
        return 'embedding'


@dataclass
class ProjectedEmbeddingsType:
    value: np.array

    @classmethod
    def name(cls):
        return 'projected_embedding'


@dataclass
class ClassProbabilitiesType:
    value: List[float]

    @classmethod
    def name(cls):
        return 'confidence'


@dataclass
class PredictedLabelsType:
    value: List[str]

    @classmethod
    def name(cls):
        return 'predicted_label'


@dataclass
class PredictedProbabilitiesType:
    value: List[np.ndarray]

    @classmethod
    def name(cls):
        return 'predicted_probabilities'


@dataclass
class LabelsForPredictedProbabilitiesType:
    value: List[List[str]]

    @classmethod
    def name(cls):
        return 'labels_predicted_probabilities'


@dataclass
class LabelsType:
    value: List[str]

    @classmethod
    def name(cls):
        return 'labels'


@dataclass
class TestSampleFlagType:
    value: bool

    @classmethod
    def name(cls):
        return 'test_sample'


@dataclass
class LabelType:
    value: str

    @classmethod
    def name(cls):
        return 'label'


@dataclass
class OoDScoreType:
    value: float

    @classmethod
    def name(cls):
        return 'ood_score'


@dataclass
class MetadataSampleType:
    relative_path: RelativePathType
    labels: LabelsType
    test_sample: TestSampleFlagType
    label: LabelType

    def to_dict(self):
        return {
            RelativePathType.name(): self.relative_path.value,
            LabelsType.name(): self.labels.value,
            TestSampleFlagType.name(): self.test_sample.value,
            LabelType.name(): self.label.value
        }
