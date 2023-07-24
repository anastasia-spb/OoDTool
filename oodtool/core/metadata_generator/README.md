# MetadataGenerator

Walks through all directories of selected dataset and generates <database_name>.meta.pkl file with columns:

Output file extension shall be ".meta.pkl"


| relative_path                 |  labels   | test_sample | label |
|-------------------------------|:---------:|------------:|------:|
| cats_test/cats_on_leash/3.jpg | Dog, Cat  |        True |   Cat |
| dogs_train/train/dog.0.jpg    | Dog, Cat  |       False |   Dog |


Example of dataset description file: https://github.com/anastasia-spb/OoDTool/tree/main/example_data/DogsCats

## Usage

```bash
$ python3 run.py -f "../../../example_data/DogsCats/description.json"
```
