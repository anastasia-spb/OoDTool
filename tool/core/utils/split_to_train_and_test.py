import os
import shutil
from sklearn.model_selection import train_test_split


def split():
    all_files = []
    images_path = '/home/vlasova/datasets/DroneBird/birds/train'
    for _, _, filenames in os.walk(images_path):
        for name in filenames:
            all_files.append(name)

    train_indices, test_indices = train_test_split(all_files, test_size=0.3, random_state=42)
    target_dir = '/home/vlasova/datasets/DroneBird/birds/test'
    for i in test_indices:
        shutil.move(os.path.join(images_path, i), os.path.join(target_dir, i))


if __name__ == "__main__":
    split()

