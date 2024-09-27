SWGE
=======
*Signed Whole-Graph Embedding methods*

* Copyright 2020-2024 Noé Cécillon *et al.*

SWGE is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation. For source availability and license information see `licence.txt`

* **Lab site:** http://lia.univ-avignon.fr/
* **GitHub repo:** https://github.com/CompNet/SWGE
* **Data:** https://doi.org/10.5281/zenodo.13851362
* **Contact:** Noé Cécillon <noe.cecillon@alumni.univ-avignon.fr>, Vincent Labatut <vincent.labatut@univ-avignon.fr>

-----------------------------------------------------------------------

If you use this source code or the associated dataset, please cite reference [[CLDA'24](#references)].

## Description
This set of scripts primarily implements the two signed whole graph embedding methods presented in our paper [[CLDA'24](#references)]: `SG2V` (Signed Graph2vec) and `WSGCN` (Whole Signed Graph Convolutional Networks). It can be used to:
* Learn the representations of whole signed graphs.
* Perform classification tasks based on the embeddings extracted with our method.

In addition, these scripts reproduce the experiments described in our paper [[CLDA'24](#references)]. In particular, they compares the performance of our method with three alternatives from the literature:
* `SiNE` [[WTAC'17](#references)]: this method handles signed graphs, but only to represent individual vertices, and not the whole graph.
* `Graph2vec` [[NCVC'17](#references)]: this method handle whole-graphs, but only for unsigned graphs.
* `SGCN` [[DMT'18](#references)]: as for SiNE, this method handles signed graphs, but only to represent individual vertices, and not the whole graph.


## Data
The scripts are meant to be applied to a corpus of three datasets constituted of signed networks annotated for graph classification. Because of GitHub's file size limit, we include only a few graphs from each dataset in the `data` folder. The full datasets can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.13851362). Place the downloaded graphs directly into the corresponding folders in `data`. 


## Organization
This repository is composed of the following elements:
* Folder `data`: input signed networks.
* Folder `out`: files produced by the scripts.
* Folder `SG2V`: implementation of our signed version of the Graph2vec method.
* Folder `SGCN`: implementation of the standard Signed Graph Convolutional Network method, taken from repository [SGCN](https://github.com/benedekrozemberczki/SGCN).
* Folder `SiNE`: implementation of the standard Signed Network Embeddings method.
* Folder `WSGCN`: implementation of our whole-graph variant of the SGCN method.
* File `requirements.txt`: list of `Python` packages used by the scripts.
* The main scripts are:
  * `main.py`: learns the graph representations.
  * `evaluation.py`: Performs the classification task.


## Installation
* Install `Python` (tested with `Python` v3.6.9)
* Install dependencies using the following command:
  ```
  pip install -r requirements.txt
  ```
* Retrieve the data from Zenodo and place them into the `data` folder (as nexplained in section [Data](#data)).


## How to use
In order to learn `Graph2Vec`-based representations, run the file `SG2V.py`. You can configure the `model_type` between the 3 versions proposed in our paper: `g2v`, `sg2vn` or `sg2vsb`. To learn representations `SGCN`-based representations, run the file `run_sgcn.py`. These scripts will export the learned representations into the `out` folder.

After running the previous scripts, you can perform the classification by running `evaluation.py`. You can configure the `path_emb` and `path_label` variables to change the dataset used. The `main.py` file can be used to run all the experiment with a single script.


## References
* **[CLDA'24]** N. Cécillon, V. Labatut, R. Dufour, N. Arınık: *Whole-Graph Representation Learning For the Classification of Signed Networks*, IEEE Access (in press), 2024. DOI: [xxxx](https://dx.doi.org/xxxx) [⟨hal-04712854⟩](https://hal.archives-ouvertes.fr/hal-04712854)
* **[NCVC'17]** A. Narayanan, M. Chandramohan, R. Venkatesan, L. Chen, Y. Liu, and S. Jaiswal: *graph2vec: Learning distributed representations of graphs*, International Workshop on Mining and Learning with Graphs, 2017. URL: [http://www.mlgworkshop.org/2017/paper/MLG2017_paper_21.pdf]
* **[DMT'18]** T. Derr, Y. Ma, and J. Tang: *Signed graph convolutional network*, 18th International Conference on Data Mining, 2018, p.929-934. DOI: [10.1109/ICDM.2018.00113](https://doi.org/10.1109/ICDM.2018.00113).
* **[WTAC'17]** S. Wang, J. Tang, C. Aggarwal, Y. Chang, and H. Liu. *Signed network embedding in social media*. 17th SIAM International Conference on Data Mining, 2017, p.327-335. DOI: [10.1137/1.9781611974973.37](https://doi.org/10.1137/1.9781611974973.37).
