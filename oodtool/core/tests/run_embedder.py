import os
import pandas as pd
from oodtool.core.towhee_adapter import TowheeAdapter


def run_embedder():
    data_dir = '/home/vlasova/datasets/ood_datasets/Letters_v20/dataset'
    ood_session_dir = '/home/vlasova/datasets/ood_datasets/Letters_v20/dataset/oodsession_3'
    output_dir = ood_session_dir

    metadata_file_path = os.path.join(ood_session_dir, 'DatasetDescription.meta.pkl')
    metadata_df = pd.read_pickle(metadata_file_path)

    # Select embedder id
    # timm_resnet50; torch_shared-regnet_trafficlights_v13; timm_swin_base_patch4_window7_224;
    # timm_swin_small_patch4_window7_224
    # emb_id = 'timm_resnet50'
    for emb_id in ['torch_shared-regnet_trafficlights_v13', 'timm_swin_base_patch4_window7_224', 'timm_swin_small_patch4_window7_224']:
        embedder_wrapper = TowheeAdapter(metadata_df, data_dir, output_dir=output_dir)
        output_file, _ = embedder_wrapper.predict(model_id=emb_id)
        print(output_file)


if __name__ == "__main__":
    run_embedder()
