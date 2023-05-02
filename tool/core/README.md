**Data Exploration Tool. Core package. **
______________________________________________________________________

***Core package includes python scripts, which help with data exploration. ***

* Model wrappers for embeddings and predictions
* * AlexNet
* * Wrappers for retrained models from `timm` deep-learning library (https://timm.fast.ai/)
* Classifiers
* Data projectors wrappers
* Nearest neighbours search using euclidian or cosine distance
* OoD Score algorithms
* * Mahalanobis
* * Entropy of the averaged probability distribution of all classifiers [[1]](#1)

______________________________________________________________________


***Data flow***

Please, see [here](././example_data/example_datasets) to see an example
how to organize your dataset for evaluation with OoDTool. The tool stores generated
data in .pkl files after each step, so you can pick up at any point. 

1. MetadataGenerator

Walk through all directories of selected dataset and generates <database_name>.meta.pkl file with columns:

| relative_path                               |          labels          | test_sample |   label |
|---------------------------------------------|:------------------------:|------------:|--------:|
| BalloonsBubbles/bubble/test/bubble_2.jpg    |     bubble, balloon      |        True |  bubble |
| BalloonsBubbles/balloon/train/balloons3.jpg |     bubble, balloon      |       False | balloon |

2. ModelWrapper

Expects .meta.pkl file. Extend table from .meta.pkl with embeddings and probabilities and stores result in
<embedder_name>_<database_name>_<timedate>.emb.pkl. 


| relative_path                               |          labels          | test_sample |   label |  embedding |                  class_probabilities |
|---------------------------------------------|:------------------------:|------------:|--------:|-----------:|-------------------------------------:|
| BalloonsBubbles/bubble/test/bubble_2.jpg    |     bubble, balloon      |        True |  bubble | np.ndarray | np.ndarray of shape(len(labels) + 1) |
| BalloonsBubbles/balloon/train/balloons3.jpg |     bubble, balloon      |       False | balloon | np.ndarray | np.ndarray of shape(len(labels) + 1) |

<span style="color:#59afe1"> Why length of class probabilities generated by embedder is greater than number of labels in provided dataset? </span>
If user selects model, which was pretrained to classify on more classes or uses other classes, then if predicted label
not in database labels, max probability will be stored at last place in probabilities array. If model and database operate with same classes, at last
place of probabilities array will be always zero.


3. ClassifierWrapper

Expects .emb.pkl. and generates following data:
* <classifier_name>_<timedate>.clf.pkl
* <classifier_name>Train folder with checkpoints

Inside .clf.pkl:


| relative_path                               |      class_probabilities_1       | ... |      class_probabilities_N       |
|---------------------------------------------|:--------------------------------:|:---:|:--------------------------------:|
| BalloonsBubbles/bubble/test/bubble_2.jpg    | np.ndarray of shape(len(labels)) | ... | np.ndarray of shape(len(labels)) |
| BalloonsBubbles/balloon/train/balloons3.jpg | np.ndarray of shape(len(labels)) | ... | np.ndarray of shape(len(labels)) |


4. OoDEntropyWrapper

Expects list of .clf.pkl files and generates .ood.pkl


| relative_path                               | ood_score |
|---------------------------------------------|:---------:|
| BalloonsBubbles/bubble/test/bubble_2.jpg    |   float   |
| BalloonsBubbles/balloon/train/balloons3.jpg |   float   |

Score in range <0.0, 1.0>

5. OoDMahalanobisWrapper

Expects .emb.pkl file and generates .ood.pkl


| relative_path                               | ood_score |
|---------------------------------------------|:---------:|
| BalloonsBubbles/bubble/test/bubble_2.jpg    |   float   |
| BalloonsBubbles/balloon/train/balloons3.jpg |   float   |

Score in range <0.0, 1.0>

6. DataProjectorWrapper

Expects .emb.pkl file and generates .2emb.pkl. It copies the content of .emb.pkl and replaces "embedding" column
with "projected_embedding".


| relative_path                               |          labels          | test_sample |   label | projected_embedding  |                  class_probabilities |
|---------------------------------------------|:------------------------:|------------:|--------:|---------------------:|-------------------------------------:|
| BalloonsBubbles/bubble/test/bubble_2.jpg    |     bubble, balloon      |        True |  bubble |           np.ndarray | np.ndarray of shape(len(labels) + 1) |
| BalloonsBubbles/balloon/train/balloons3.jpg |     bubble, balloon      |       False | balloon |           np.ndarray | np.ndarray of shape(len(labels) + 1) |


7. DistanceWrapper

Expects .emb.pkl file and generates .dist.pkl. In columns from 1 to N are stored distance to objects
in corresponding rows. 


| relative_path                               |   1   |   2   | ... |   N   |
|---------------------------------------------|:-----:|:-----:|:---:|:-----:|
| BalloonsBubbles/bubble/test/bubble_2.jpg    |  0.0  | float | ... | float |
| BalloonsBubbles/balloon/train/balloons3.jpg | float |  0.0  | ... | float |


## References
<a id="1">[1]</a> 
Changjian Chen and Jun Yuan and Yafeng Lu and Yang Liu and Hang Su and Songtao Yuan and Shixia Liu (2020). 
OoDAnalyzer: Interactive Analysis of Out-of-Distribution Samples. https://arxiv.org/abs/2002.03103