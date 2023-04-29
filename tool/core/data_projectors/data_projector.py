import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import trimap
import umap.umap_ as umap

from tool.core.data_types import types

from tool.core.data_projectors.pca_tsne_projector import PcaTsneProjector


class DataProjector:
    methods = ['pca_tsne', 'tsne', 'pca', 'mds', 'trimap', 'umap']
    method_name = methods[0]

    def __init__(self, method_name, verbose=0):
        self.projector = None
        self.method_name = method_name
        self.projected_data = None
        if method_name == "tsne":
            self.projector = TSNE(n_components=2, perplexity=30.0, n_iter=2000, learning_rate=10.0,
                                  random_state=42, verbose=verbose)
        elif method_name == "pca":
            self.projector = PCA(n_components=2, svd_solver='full', random_state=42)
        elif method_name == "mds":
            self.projector = MDS(n_components=2, n_jobs=-1, random_state=42, verbose=1)
        elif method_name == "pca_tsne":
            self.projector = PcaTsneProjector(n_components=2, verbose=verbose)
        elif method_name == "trimap":
            self.projector = trimap.TRIMAP()
        elif method_name == "umap":
            self.projector = umap.UMAP()

    def __fit_transform(self, X):
        self.projected_data = self.projector.fit_transform(X)
        return self.projected_data

    def project(self, metadata_folder, embeddings_file):
        data_df = pd.read_pickle(embeddings_file)
        X = data_df[types.EmbeddingsType.name()].tolist()
        embeddings = np.array(X, dtype=np.dtype('float64'))
        x = self.__fit_transform(embeddings)
        data_df.drop([types.EmbeddingsType.name()], axis=1, inplace=True)
        data_df[types.ProjectedEmbeddingsType.name()] = x.tolist()

        timestamp_str = datetime.now().strftime("%y%m%d_%H%M%S")
        name = "".join((self.method_name, "_", timestamp_str, ".2emb.pkl"))
        output_file = os.path.join(metadata_folder, name)
        data_df.to_pickle(output_file)
        return output_file



