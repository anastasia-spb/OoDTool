import os
import pandas as pd
from oodtool.core.data_types import types


def get_ood_file(folder):
    for file in os.listdir(folder):
        if file.endswith(".ood.pkl"):
            return os.path.join(folder, file)
    return ''


def get_2d_emb_file(folder):
    for file in os.listdir(folder):
        if file.endswith(".2emb.pkl"):
            return os.path.join(folder, file)
    return ''


def get_metadata_file(folder):
    for file in os.listdir(folder):
        if file.endswith(".meta.pkl"):
            return os.path.join(folder, file)
    return ''


def get_classes_from_metadata_file(folder):
    for file in os.listdir(folder):
        if file.endswith(".meta.pkl"):
            data_df = pd.read_pickle(os.path.join(folder, file))
            labels = data_df[types.LabelsType.name()][0]
            return labels
    return []


def get_embeddings_file(folder):
    for file in os.listdir(folder):
        if file.endswith(".emb.pkl"):
            return os.path.join(folder, file)
    return ''


def get_embeddings_files(folder):
    files = []
    for file in os.listdir(folder):
        if file.endswith(".emb.pkl"):
            files.append(os.path.join(folder, file))
    return files


def get_distances_file(folder):
    for file in os.listdir(folder):
        if file.endswith(".dist.pkl"):
            return os.path.join(folder, file)
    return ''


def get_description_file(folder):
    for file in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file)) and file.endswith(".desc.json"):
            return os.path.join(folder, file)
    return None
