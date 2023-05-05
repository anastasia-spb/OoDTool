## Classifiers

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

Please, see ...

## LogisticRegressionWrapper parameters




## LinearClassifierWrapper parameters

