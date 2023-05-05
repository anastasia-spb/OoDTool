# Classifiers

OoDTools implements two types of classifier ensembles. First one consists of 
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">sklearn.linear_model.LogisticRegression</a> classifiers.
In the multiclass case, the training algorithm uses the one-vs-rest approach.
Second one groups classifiers built with single <a href="https://pytorch.org/docs/stable/generated/torch.nn.Linear.html"> linear layer </a>.
Layer output is transformed to probabilities by <a href="https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html"> softmax function </a>.


## Usage

Classifiers training pipeline can be launched via OoDTool gui or directly from terminal:

```bash
$ python3 run.py -emb "features_1.emb.pkl" "features_2.emb.pkl" "features_n.emb.pkl" -probs "features_2.emb.pkl"
```

Further user will be asked to select classifier type and enter parameters.
For every feature file will be combined with each classifier. Results and pretrained weights will be stored 
besides feature file. 

## Output format

For every feature file (with .emb.pkl extension) will be generated one .clf.pkl file with following columns:   

| relative_path                               |      class_probabilities_1       | ... |      class_probabilities_N       |
|---------------------------------------------|:--------------------------------:|:---:|:--------------------------------:|
| BalloonsBubbles/bubble/test/bubble_2.jpg    | np.ndarray of shape(len(labels)) | ... | np.ndarray of shape(len(labels)) |
| BalloonsBubbles/balloon/train/balloons3.jpg | np.ndarray of shape(len(labels)) | ... | np.ndarray of shape(len(labels)) |



## Parameters

Different classification results for potential OoD samples are obtained by varying regularization strength.
Regularization is one of the common approaches to avoid overfitting - by preventing any particular weight from growing too high.

ScikitLearn Logistic Regression uses L2 Regularization by default (so OoD tool).
For LR Classifier ensemble you shall vary parameter `C`, which is inverse of regularization strength.
Smaller values specify stronger regularization. It's recommended to select values from {10e-6 to 10e6} with step
proportional to the amount of classifiers inside ensemble to better cover the parameter space.
For more details, please see [[1]](#1)

LinearClassifier implementation differs from LogisticRegression approach in two major ways:
it can be used for multilabel classification and uses stochastic gradient descent 
<a href="https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html"> SGD </a> learning.
Regularization strength is controlled by `weight_decay` parameter. 
The parameter for weight decay is set on a logarithmic scale between 0 and 0.1 (0.1, 0.01, 0.001, ...).
L2 regularization and weight decay regularization are equivalent for standard stochastic gradient descent 
(when rescaled by the learning rate). 

When experimenting with adaptive gradient optimizers, such as Adam or AdamW, weight decay parameter
shall be selected from different range.


https://pytorch.org/docs/stable/generated/torch.optim.SGD.html

## References
<a id="1">[1]</a> 
Changjian Chen and Jun Yuan and Yafeng Lu and Yang Liu and Hang Su and Songtao Yuan and Shixia Liu (2020). 
OoDAnalyzer: Interactive Analysis of Out-of-Distribution Samples. https://arxiv.org/abs/2002.03103