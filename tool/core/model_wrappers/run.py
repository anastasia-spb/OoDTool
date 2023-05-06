import os
import argparse
from typing import List

from tool.core.model_wrappers.models.timm_resnet.timm_resnet_wrapper import TimmResnetWrapper, SUPPORTED_CHECKPOINTS
from tool.core.model_wrappers.embedder_pipeline import EmbedderPipeline
from tool.core.model_wrappers.embedder_pipeline import MODEL_WRAPPERS
from tool.core.model_wrappers.models.timm_resnet.imagenet1000_clsidx_to_labels import ImageNetLabelsTag


def run():
    parser = argparse.ArgumentParser(description='Generate embeddings')
    parser.add_argument('-meta', '--metadata_file', default='', required=True)
    parser.add_argument('-d', '--dataset_dir', default='', required=True)
    parser.add_argument('--grads', action='store_true')
    parser.set_defaults(grads=False)

    print("Select model wrapper from {0}".format(MODEL_WRAPPERS.keys()))
    embedder_name = input()

    model = ''
    if embedder_name == TimmResnetWrapper.get_name():
        print("Select model from {0}".format(SUPPORTED_CHECKPOINTS))
        model = input()

    labels = ImageNetLabelsTag
    print("If you use pretrained model on other than ImageNet dataset, please, enter labels for"
          "meaningful probabilities in format {0}. Otherwise press ENTER".format('label1, label2, label3'))
    user_labels = input()
    if user_labels != '':
        labels = user_labels

    args = parser.parse_args()

    output_folder, _ = os.path.split(args.metadata_file)

    print("Enter path to checkpoints or press ENTER if you selected pretrained model type: ")
    checkpoint_path = input()

    wrapper_parameters = {"checkpoint_path": checkpoint_path,
                          "model_labels": labels,
                          "model_checkpoint": model}

    pipeline = EmbedderPipeline(args.metadata_file, args.dataset_dir, embedder_name,
                                use_cuda=True, **wrapper_parameters)

    def callback(progress_info: List[int]):
        pass

    pipeline.predict(callback, requires_grad=args.grads)


if __name__ == "__main__":
    run()
