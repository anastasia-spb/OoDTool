# Distance between embeddings

This module provides functionality for calculation pairwise distances between samples.

## Usage

```bash
$ python3 run.py -emb "features.emb.pkl" -m "cosine"
```


## Output format

Expects .emb.pkl file and generates .dist.pkl. In columns from 1 to N are stored distance to objects
in corresponding rows. 


| relative_path                               |   1   |   2   | ... |   N   |
|---------------------------------------------|:-----:|:-----:|:---:|:-----:|
| BalloonsBubbles/bubble/test/bubble_2.jpg    |  0.0  | float | ... | float |
| BalloonsBubbles/balloon/train/balloons3.jpg | float |  0.0  | ... | float |
