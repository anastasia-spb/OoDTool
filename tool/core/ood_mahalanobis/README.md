# OoD Score (Mahalanobis)


## Usage

Use GT for training:

```bash
$ python3 run.py -emb <features>.emb.pkl
```

Use probabilities for training:

```bash
$ python3 run.py -emb <features>.emb.pkl -probs <features_with_probabilities>.emb.pkl
```

## Output format

Creates .ood.pkl file with columns: 

| relative_path                               | ood_score |
|---------------------------------------------|:---------:|
| BalloonsBubbles/bubble/test/bubble_2.jpg    |   float   |
| BalloonsBubbles/balloon/train/balloons3.jpg |   float   |

Score in range <0.0, 1.0>
