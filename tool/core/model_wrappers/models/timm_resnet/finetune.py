import pandas as pd
import gc
import torch

from tool.core.utils.mock_missing import mock_missing

try:
    from fastai.vision.all import vision_learner, ImageDataLoaders, accuracy, Resize
except ImportError:
    vision_learner = mock_missing('vision_learner')
    ImageDataLoaders = mock_missing('ImageDataLoaders')
    accuracy = mock_missing('accuracy')
    Resize = mock_missing('Resize')

from tool.core import data_types


def finetune():
    """
    Fine tune selected timm model with fastai lib
    https://timm.fast.ai/#Fine-tune-timm-model-in-fastai
    """

    # Cuda maintenance
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_root = '/home/vlasova/Desktop/NIR/NIR/OoDTool/example_data/DogsCats'
    path_to_meta = '/home/vlasova/Desktop/NIR/NIR/OoDTool/example_data/DogsCats/oodsession_0/DogsCats.meta.pkl'
    meta_df = pd.read_pickle(path_to_meta)

    labels = meta_df[data_types.LabelsType.name()][0]
    meta_df[data_types.LabelType.name()] = \
        meta_df.apply(lambda row: labels.index(row[data_types.LabelType.name()]), axis=1)

    dls = ImageDataLoaders.from_df(meta_df, path=dataset_root, fn_col=data_types.RelativePathType.name(),
                                   valid_col=data_types.TestSampleFlagType.name(), bs=8,
                                   label_col=data_types.LabelType.name(), item_tfms=Resize(224), device=device)

    model = 'densenet121'
    epochs_to_finetune = 1
    learn = vision_learner(dls, model, n_out=len(labels), metrics=accuracy)
    learn.fine_tune(epochs_to_finetune)

    learn.save('fine_tuned_{0}_epochs_{1}'.format(model, epochs_to_finetune))


if __name__ == "__main__":
    finetune()
