import json
import os.path

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
    value: np.array

    @classmethod
    def name(cls):
        return 'class_probabilities'


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
