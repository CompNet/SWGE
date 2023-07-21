# SWGE
Our implementation of Signed Whole Graph Embeddings methods.

## Description
This set of scripts implements the signed whole graph embedding methods presented in our paper. 
It can be used to:
* Learn the representations of whole signed graphs.
* Perform classification tasks based on the embeddings extracted from signed methods.

## Data
We include a few graphs for each dataset in the `data` folder. The full datasets can be downloaded from [anonymity - available later] Place the downloaded graphs directly into the corresponding folder in `data`. 

## Organization

This repository is composed of the following elements:

* Folder `data`: input signed networks.
* Folder `out`: contains the files produced by our scripts.

* `requirements.py`: List of Python packages used in SWGE.

* There are 3 main scripts:
  * `SG2V.py`: Performs the representation learning step related to the Signed Graph2Vec methods.
  * `run_sgcn.py`: Performs the representation learning step related to the Signed Graph Convolutional Networks methods.
  * `evaluation.py`: Performs the classification task.


## Installation

* Install Python (tested with Python 3.6.9)
* Install dependencies using the following command:
  ```
  pip install -r requirements.txt
  ```
* In order to use the SGCN method, you have to download the implementation from [SGCN](https://github.com/benedekrozemberczki/SGCN) and place it inside a `SGCN-master` folder.
* Retrieve the data from Zenodo and place them into the `data` folder.

## How to use ?
* In order to learn representations with Signed Graph2Vec, run the file `SG2V.py`. You can configure the `model_type` between the 3 versions proposed in our paper: `g2v`, `sg2vn` or `sg2vsb`.
* In order to learn representations with SGCN, run the file `run_sgcn.py`.

These scripts will export the learned representations into the `out` folder.

* After running the previous scripts, you can perform the classification by running `evaluation.py`. You can configure the `path_emb` and `path_label` variables to change the dataset used.


## References 
* A. Narayanan, M. Chandramohan, R. Venkatesan, L. Chen, Y. Liu, and S. Jaiswal: *graph2vec: Learning distributed representations of graphs*, MLG, 2017. URL: [http://www.mlgworkshop.org/2017/paper/MLG2017_paper_21.pdf]
* T. Derr, Y. Ma, and J. Tang: *Signed graph convolutional network*, 18th ICDM, 2018, DOI: [10.1109/ICDM.2018.00113](https://doi.org/10.1109/ICDM.2018.00113).
