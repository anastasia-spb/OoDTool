import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from tool.core import data_types


class JpegDataset(Dataset):
    def __init__(self, metadata: str, root_dir: str, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.img_df = pd.read_pickle(metadata)
        self.root_dir = root_dir
        if self.img_df.shape[0] > 0:
            self.labels = self.img_df.labels[0]
        else:
            self.labels = []

    def __label_to_idx(self, label: str):
        return self.labels.index(label)

    def get_metadata(self):
        return self.img_df

    def __len__(self):
        return self.img_df.shape[0]

    def __getitem__(self, index):
        img_metadata = self.img_df.iloc[index]
        label = self.__label_to_idx(img_metadata[data_types.LabelType.name()])
        img_name = os.path.join(self.root_dir, img_metadata[data_types.RelativePathType.name()])
        with Image.open(img_name).convert('RGB') as image:
            if self.transform is not None:
                image = self.transform(image)
        return image, label, img_metadata[data_types.RelativePathType.name()]


class JpegTrainDataset(JpegDataset):
    def __init__(self, metadata: str, root_dir: str, transform=None):
        super().__init__(metadata, root_dir, transform)
        self.img_df = self.img_df.loc[self.img_df.test_sample == False].reset_index(drop=True)


class JpegTestDataset(JpegDataset):
    def __init__(self, metadata: str, root_dir: str, transform=None):
        super().__init__(metadata, root_dir, transform)
        self.img_df = self.img_df.loc[self.img_df.test_sample == True].reset_index(drop=True)
