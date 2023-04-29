import os
import pandas as pd
import numpy as np
from PIL import Image

from tool.qt_utils import find_pkl
from tool.qt_utils.qt_types import ImageInfo
from tool.core import data_types


def get_k_neighbours(info: ImageInfo, k: int):
    images_paths = []
    # TODO: move file reading into outer class
    distances_pkl = find_pkl.get_distances_file(info.metadata_dir)
    if not os.path.isfile(distances_pkl):
        return images_paths

    distances_df = pd.read_pickle(distances_pkl)
    img_index = distances_df.index[distances_df[data_types.RelativePathType.name()] == info.relative_path]
    img_row = distances_df.iloc[img_index]
    img_row = img_row.drop(columns=[data_types.RelativePathType.name()])
    img_row = img_row.values.tolist()
    distances = np.array(img_row, dtype=np.dtype('float64'))
    smallest_dist_indices = np.argsort(np.absolute(distances[0]))[:k]

    images_paths = [distances_df[data_types.RelativePathType.name()][i] for i in smallest_dist_indices]
    images_meta = [ImageInfo(path=path, score=-1.0, probs=[-1.0, -1.0], labels=info.labels,
                             absolute_path=info.absolute_path, metadata_dir=info.metadata_dir) for path in images_paths]

    return images_meta


def test():
    info = ImageInfo(path='DroneBird/birds_test/fdc4966a-559b-4e93-9c56-98b86dacdc0d.jpg', score=0.0, probs=[1.0, 0.0],
                     labels=["Drone", "Bird"],
                     absolute_path='/home/vlasova/datasets', metadata_dir='/home/vlasova/datasets/0metadata/DroneBird')
    images_meta = get_k_neighbours(info, k=4)
    img = Image.open(os.path.join(info.absolute_path, info.relative_path))
    img.show()

    for neighbour_meta in images_meta:
        print(neighbour_meta.relative_path)
        neighbour_img = Image.open(os.path.join(neighbour_meta.absolute_path, neighbour_meta.relative_path))
        neighbour_img.show()


if __name__ == "__main__":
    test()
