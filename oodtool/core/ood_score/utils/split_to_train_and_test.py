from sklearn.model_selection import train_test_split
from typing import List, Callable
import numpy as np
import math


def filter_small_sets(y_train: List[str], logs_callback: Callable[[str], None] = None):
    unique, counts = np.unique(y_train, return_counts=True)
    labels = []
    corresponding_counts = []
    for tag, n in zip(unique, counts):
        if logs_callback is not None:
            logs_callback(f"Class {tag} contains {n}")
        if n > 1:
            labels.append(tag)
            corresponding_counts.append(n)
        else:
            if logs_callback is not None:
                logs_callback(f"Class {tag} with {n} sample dropped")
    return labels, corresponding_counts


def get_train_indices(y_train: List[str], y_train_cat, logs_callback: Callable[[str], None] = None,
                      max_num_training_samples: int = 3000):
    labels, samples_count = filter_small_sets(y_train, logs_callback)
    max_training_samples_num = max(max_num_training_samples, math.ceil(len(y_train) / min(samples_count)))

    all_train_indices = [index for (index, item) in enumerate(y_train) if item in labels]

    if logs_callback is not None:
        logs_callback(f"{len(all_train_indices)} samples selected for training")

    if len(all_train_indices) <= max_training_samples_num:
        return all_train_indices, len(labels)

    train_indices, _ = train_test_split(all_train_indices, train_size=max_training_samples_num,
                                        random_state=42, shuffle=True,
                                        stratify=np.take(y_train_cat, all_train_indices))
    return train_indices, len(labels)
