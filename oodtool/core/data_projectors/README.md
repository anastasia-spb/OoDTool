# Project data embeddings to 2D plane

This module is a wrapper for mostly used dimensionality reduction methods:

1. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html"> t-SNE </a>
2. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA"> PCA </a>
3. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS"> MDS </a>
4. <a href="https://pypi.org/project/trimap/"> TriMap </a>
5. <a href="https://umap-learn.readthedocs.io/en/latest/"> UMAP </a>
6. pca_tsne: It's a combination of<a href=" https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html"> IncrementalPCA </a>
and t-SNE. For faster calculation.

## Usage

```bash
$ python3 run.py -emb "features.emb.pkl" -m "tsne"
```


## Output format

Expects .emb.pkl file and generates .2emb.pkl. It copies the content of .emb.pkl and replaces "embedding" column
with "projected_embedding".


| relative_path                               |     projected_embedding |
|---------------------------------------------|------------------------:|
| BalloonsBubbles/bubble/test/bubble_2.jpg    |              np.ndarray |
| BalloonsBubbles/balloon/train/balloons3.jpg |              np.ndarray | 
