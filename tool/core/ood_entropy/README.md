# OoD Score (Entropy)

The OoD score of each sample is the entropy of the averaged probability distribution of all classifiers [[1]](#1)

## Usage

```bash
$ python3 run.py -p "probabilities_1.clf.pkl" "probabilities_2.clf.pkl" "probabilities_n.clf.pkl"
```


## Output format

Creates .ood.pkl file with columns: 

| relative_path                               | ood_score |
|---------------------------------------------|:---------:|
| BalloonsBubbles/bubble/test/bubble_2.jpg    |   float   |
| BalloonsBubbles/balloon/train/balloons3.jpg |   float   |

Score in range <0.0, 1.0>

## References
<a id="1">[1]</a> 
Changjian Chen and Jun Yuan and Yafeng Lu and Yang Liu and Hang Su and Songtao Yuan and Shixia Liu (2020). 
OoDAnalyzer: Interactive Analysis of Out-of-Distribution Samples. https://arxiv.org/abs/2002.03103