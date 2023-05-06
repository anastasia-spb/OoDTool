# Project data embeddings to 2D plane



## Usage

```bash
$ python3 run.py -emb "features.emb.pkl" -m "tsne"
```


## Output format

Expects .emb.pkl file and generates .2emb.pkl. It copies the content of .emb.pkl and replaces "embedding" column
with "projected_embedding".


| relative_path                               |          labels          | test_sample |   label | projected_embedding  |                  class_probabilities |
|---------------------------------------------|:------------------------:|------------:|--------:|---------------------:|-------------------------------------:|
| BalloonsBubbles/bubble/test/bubble_2.jpg    |     bubble, balloon      |        True |  bubble |           np.ndarray | np.ndarray of shape(len(labels) + 1) |
| BalloonsBubbles/balloon/train/balloons3.jpg |     bubble, balloon      |       False | balloon |           np.ndarray | np.ndarray of shape(len(labels) + 1) |
