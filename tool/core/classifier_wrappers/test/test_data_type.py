from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestData:
    classifier_tag: str
    embeddings_pkl: List[str]
    pkl_with_probabilities: Optional[str]
    kwargs: List[dict]
    use_gt: bool
