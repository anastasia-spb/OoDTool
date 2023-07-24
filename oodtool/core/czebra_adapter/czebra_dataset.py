from PIL import Image
import os
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset

from oodtool.core import data_types
from oodtool.core.utils.data_helpers import label_to_idx


class CZebraDataset(IterableDataset):
    def __init__(self, img_df: pd.DataFrame, root_dir: str):
        super(IterableDataset, self).__init__()
        self.img_df = img_df
        self.root_dir = root_dir
        if self.img_df.shape[0] > 0:
            self.labels = self.img_df.labels[0]
        else:
            self.labels = []

    def get_labels(self):
        return self.labels

    def __label_to_idx(self, label: str):
        return label_to_idx(self.labels, label)

    def get_metadata(self):
        return self.img_df

    def __len__(self):
        return self.img_df.shape[0]

    def __iter__(self):
        convert_row = lambda row: self.__getitem__(row[0])
        return map(convert_row, self.img_df.iterrows())

    def __read_img(self, path_image: str):
        pil_image = Image.open(path_image).convert('RGB')
        img = np.array(pil_image)
        # Convert RGB to BGR
        return img[:, :, ::-1].copy()

    def __getitem__(self, index):
        img_metadata = self.img_df.iloc[index]
        label = self.__label_to_idx(img_metadata[data_types.LabelType.name()])
        img_name = os.path.join(self.root_dir, img_metadata[data_types.RelativePathType.name()])
        image = self.__read_img(img_name)
        return image, label, img_metadata[data_types.RelativePathType.name()]
