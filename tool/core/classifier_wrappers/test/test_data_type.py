from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestData:
    classifier_tag: str
    embeddings_pkl: str
    pkl_with_probabilities: Optional[str]
    weight_decays: List[float]
    use_gt: bool
    checkpoint: str
