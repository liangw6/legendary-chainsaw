## Requirement
Uses libraries Pandas, Numpy and Networkx.

You can also install tqdm for progress bar during future steps

## Usage
Download data from stack overflow 10% (https://www.kaggle.com/stackoverflow/stacksample#Tags.csv).

Extract Answers.csv into `./data/ directory`

`stack-overflow-graph-processing.ipynb` will pre-process the data, build graph and run random walk algorithm on a single node

Note:

* be sure to also create a `./processed_data/` directory, where the edges for the graph will be stored and read

* Using pandas to load the stack overflow data eats up ~2GB of memory. But after pre-processing, the rest of the variables are more light-weight.

TODO: Evaluation. 

