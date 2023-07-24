import os
import pandas as pd
from typing import List
from oodtool.core.ood_score.ood_entropy.score_embeddings import score_embeddings
from oodtool.core.ood_score import ood_score_to_df


def run_from_checkpoints(checkpoints_root_dir: str, checkpoints_files: List[str]):
    assert len(checkpoints_files) > 0

    metadata_folder = './test_data/pedestrian_tl/ood_session_test/'
    embeddings_files = [os.path.join(metadata_folder, 'torch_embedder_towheeresnet50_v2.emb.pkl')]
    metadata_file = os.path.join(metadata_folder, 'DatasetDescription.meta.pkl')
    metadata_df = pd.read_pickle(metadata_file)

    checkpoints = [os.path.join(checkpoints_root_dir, f) for f in checkpoints_files]

    score = score_embeddings(embeddings_files, metadata_df, checkpoints=[checkpoints])

    if score is not None:
        file_name = "".join(("ood_entropy_from_checkpoints", '.ood.pkl'))
        full_path = os.path.join(metadata_folder, file_name)
        df = ood_score_to_df(score, metadata_df)
        df.to_pickle(full_path)
        print(full_path)
