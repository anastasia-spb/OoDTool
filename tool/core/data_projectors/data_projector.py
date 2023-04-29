import os
import numpy as np
from datetime import datetime
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import trimap
import umap.umap_ as umap

from tool.data_projectors.pca_tsne_projector import PcaTsneProjector


class DataProjector(object):
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

    def fit_transform(self, X):
        self.projected_data = self.projector.fit_transform(X)
        return self.projected_data


