## Requirement
This majority of feature extraction uses sentence-bert, which relies on python 3.6, PyTorch and transformers. See https://github.com/UKPLab/sentence-transformers for installation guide.

To install PyTorch, see https://pytorch.org/. Note that if you want to use PyTorch with GPU, all you need is an NVIDIA GPU and a corresponding driver. PyTorch comes with CUDA, so you don't need to manually install that.

Alternatively, you could also use pytorch with cpu only (see pytorch website instructions). It might be a little slower though.

The feature extraction also uses PySpark to examine and analyze the data. `pip3 install pyspark` should work.

## Usage
build_graph.ipynb:

* reads in the quora question pair document and converts each sentence to a corresponding numpy vector

* You can try directly loading the feature vectors from the disk, i.e. the second to last cell in the jupyter notebook. This should work...

