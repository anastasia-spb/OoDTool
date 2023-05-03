from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestData:
    classifier_tag: str
    embeddings_pkl: str
    pkl_with_probabilities: Optional[str]
    kwargs: List[dict]
    use_gt: bool
    checkpoint: None
