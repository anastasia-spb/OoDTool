# OoD Score (Mahalanobis)

Alternative approach to calculate OoD Score.
This method requires to be trained on embeddings and labels of in-distribution samples.

For each cluster parameters of <a href="https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html"> Gaussian mixture distribution </a> are estimated.
OoD Score for each sample is calculated as 
<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html#scipy.spatial.distance.mahalanobis"> mahalanobis </a> distance
between test sample and the closest class-conditional Gaussian distribution.

All calculated scores are passed though softmax function to get probabilities values between 0 and 1.


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

```bibtex
@misc{ren2021simple,
      title={A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection}, 
      author={Jie Ren and Stanislav Fort and Jeremiah Liu and Abhijit Guha Roy and Shreyas Padhy and Balaji Lakshminarayanan},
      year={2021},
      eprint={2106.09022},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```