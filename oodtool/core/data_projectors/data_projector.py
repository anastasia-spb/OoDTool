import os
import numpy as np
import pandas as pd
from oodtool.core.data_types import types


class DataProjector:
    methods = ['pca_tsne', 'tsne', 'pca', 'trimap', 'umap']
    method_name = methods[0]

    def __init__(self, method_name, verbose=0):
        self.projector = None
        self.method_name = method_name
        self.projected_data = None
        self.output_file = ''
        if method_name == "tsne":
            from sklearn.manifold import TSNE
            self.projector = TSNE(n_components=2, perplexity=30.0, n_iter=2000, learning_rate=10.0,
                                  random_state=42, verbose=verbose)
        elif method_name == "pca":
            from sklearn.decomposition import PCA
            self.projector = PCA(n_components=2, svd_solver='full', random_state=42)
        elif method_name == "pca_tsne":
            from oodtool.core.data_projectors.pca_tsne_projector import PcaTsneProjector
            self.projector = PcaTsneProjector(n_components=2, verbose=verbose)
        elif method_name == "trimap":
            import trimap
            self.projector = trimap.TRIMAP()
        elif method_name == "umap":
            import umap.umap_ as umap
            self.projector = umap.UMAP()

    def __fit_transform(self, X):
        self.projected_data = self.projector.fit_transform(X)
        return self.projected_data

    def project(self, metadata_folder, embeddings_file):
        data_df = pd.read_pickle(embeddings_file)
        X = data_df[types.EmbeddingsType.name()].tolist()
        embeddings = np.array(X, dtype=np.dtype('float64'))
        x = self.__fit_transform(embeddings)

        # Store results
        results_df = pd.DataFrame()
        results_df[types.RelativePathType.name()] = data_df[types.RelativePathType.name()]
        results_df[types.ProjectedEmbeddingsType.name()] = x.tolist()

        _, name_base = os.path.split(embeddings_file)
        name_base, _ = os.path.splitext(name_base)
        name = "".join((self.method_name, "_", name_base, ".2emb.pkl"))
        self.output_file = os.path.join(metadata_folder, name)
        results_df.to_pickle(self.output_file)
        return self.output_file

    def get_output_file(self):
        return self.output_file



