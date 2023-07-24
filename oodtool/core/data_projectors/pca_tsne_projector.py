from scipy import sparse
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE


class PcaTsneProjector:

    def __init__(self, n_components, verbose=0):
        self.n_components = n_components
        self.verbose = verbose

    def _pca(self, embedding_list):
        # batch_size is inferred from the data and set to 5 * n_features
        # n_components is set to min(n_samples, n_features)
        ipca = IncrementalPCA(n_components=None, batch_size=None)
        X_sparse = sparse.csr_matrix(embedding_list)
        return ipca.fit_transform(X_sparse)

    def fit_transform(self, X):
        pca = self._pca(X)

        _tsne = TSNE(n_components=self.n_components,
                     verbose=self.verbose,
                     n_iter=3000,
                     learning_rate='auto',
                     perplexity=30,
                     n_jobs=-1)

        return _tsne.fit_transform(pca)
