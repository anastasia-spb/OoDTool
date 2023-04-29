import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE


class PcaTsneProjector:
    """
    Copied from
    https://gitlab.cognitivepilot.com/ml/apps/data-selector/-/blob/master/data_selector/widgets/map_widget/tools/data_getter.py#L296
    """
    def __init__(self, n_components, verbose=0):
        self.n_components = n_components
        self.verbose = verbose

    def _pca(self,
             embedding_list):

        batch_size = 200
        n_samples = embedding_list.shape[0]

        ipca = IncrementalPCA(
            # n_components=200
        )

        number_of_batches = int(n_samples / batch_size)

        for i in range(number_of_batches):
            ipca.partial_fit(embedding_list[
                             slice(i * batch_size, (i + 1) * batch_size), :])

        pca = ipca.fit_transform(embedding_list)

        for i in range(len(pca)):
            pca[i] = pca[i] / np.linalg.norm(pca[i])

        return pca

    def fit_transform(self, X):
        pca = self._pca(X)

        n_iter_ = 3000

        perplexity_ = min(700,
                          max(50, int(len(pca) / 100))
                          )
        learning_rate_ = min(5000,
                             max(500, int(len(pca) / 12)))

        _tsne = TSNE(n_components=self.n_components,
                     verbose=self.verbose,
                     n_iter=n_iter_,
                     learning_rate=learning_rate_,
                     perplexity=perplexity_,
                     n_jobs=-1)
        return _tsne.fit_transform(pca)
