# Distance between embeddings

This module allows to select samples from dataset by applying density biased sampling:
under-sample dense regions and over-sample light regions.

Implemented algorithm calculates probability for each sample as following:
ood_score * confidence * density_score.

Such score was chosen in order to prefer selection of samples from under-sample dense regions
with high ood score and confidence.

## Usage

```bash
$ python3 run.py -emb <features>.emb.pkl -ood <score>.ood.pkl -n 300
```


## Output format

Copies selected samples from input .emb.pkl file into .sampled.emb.pkl file.


```bibtex
@article{article,
author = {Palmer, Christopher and Faloutsos, Christos},
year = {2000},
month = {04},
pages = {},
title = {Density Biased Sampling: An Improved Method for Data Mining and Clustering},
volume = {29},
journal = {ACM SIGMOD Record}
}
```