<div align="center">

<img src="readme_data/ood_logo_v3.png" width="400px">

**Data Exploration Tool**
______________________________________________________________________

## Getting Started

</div>

### Ubuntu


```bash
python3 -m venv ood_env

source ood_env/bin/activate

pip install --upgrade pip

pip install -e . 
```

_Troubleshooting_
  > * Error: Could not load the Qt platform plugin "xcb"
  >   Solution: 
  >   ```bash
  >   pip3 uninstall opencv-python
  >   pip3 install --no-binary opencv-python opencv-python
  >   ```

### Windows

```bash
pip install virtualenv

virtualenv --python C:\Path\To\Python\python.exe ood_env

.\ood_env\Scripts\activate

pip install --upgrade pip

pip install -e . 

```

_Troubleshooting_
  > * Error: Could not run 'aten::*' with arguments from the 'CUDA' backend.
  >   Solution - use CPU: 
  >   ```bash
  >   pip install -e . --extra-index-url "https://download.pytorch.org/whl/cpu"
  >   ```

Example dataset can be found [here](./example_data/DogsCats)

```bash
python -m oodtool
```

### Notes

For public version saliency map generation and traffic lights embedder are not available.


![](readme_data/tool.png)

![](readme_data/demo.gif)