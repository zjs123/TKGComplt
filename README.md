# pytorch implementation of A Two-Phase Framework for Temporal-Aware Knowledge Graph Completion 

Paper: [A Two-Phase Framework for Temporal-Aware Knowledge Graph Completion](https://arxiv.org/abs/1904.05530)

We propose a novel two-phase model to infer missing facts in temporal knowledge graph by utilizing temporal information, and this repository contains the implementation of our two-phase model described in the paper.

<p align="center"><img src="figs/renet.png" width="500"/></p>

Temporal knowledge graph (TKG) completion task aims to add newfacts to a KG by making inferences from facts contained in the existing triples andinformation of valid time. Many methods has been proposed towards this prob-lem, but most of them ignore temporal rules which is accurate and explainable fortemporal KG completion task. In this paper, we present a novel two-phase frame-work which integrate the advantage between time-aware embedding and temporal rules. Firstly, a trans-based temporal KG representation method is proposed tomodel the semantic information and temporal information of KG. Then a refinement model is utilized to further improve the performance of current task, which is achieved by solving a joint optimizing problem as an integer linear programming  (ILP).  

If you make use of this code in your work, please cite the following paper:

```
@inproceedings{salvador2017learning,
  title={A Two-Phase Framework for Temporal-Aware Knowledge Graph Completion},
  author={Salvador, Amaia and Hynes, Nicholas and Aytar, Yusuf and Marin, Javier and 
          Ofli, Ferda and Weber, Ingmar and Torralba, Antonio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```

## Contents
1. [Installation](#installation)
2. [Train_and_Test](#Train_and_Test)
3. [Datasets](#Datasets)
4. [Baselines](#Baselines)
5. [Contact](#contact)

## Installation

Install the following packages:

```
pip install torch
pip install numpy
```

Install CUDA and cudnn. Then run:

```
pip install cutorch
pip install cunn
pip install cudnn
```

Then clone the repository::

```
git clone https://github.com/shengyp/TKGComplt.git
```

We use Python3 for data processing and our code is also written in Python3. 

## Train_and_Test

Before running, the user should process the datasets at first
```
cd datasets/DATA_NAME
python data_processing.py
```
Then, Train the model
```
cd ..
python Train.py -d
```
Finally, ILP model was used to predict new facts
```
python ILP_solver.py
```
The default hyperparameters give the best performances.

## Datasets

There are three datasets used in our experiment:YAGO11K, WIKIDATA12K and WIKIDATA36K. facts of each datases have time annotation, which is formed as "[start_time , end_time]". Each data folder has six files: 

-entity2id.txt: the first column is entity name, and second column is index of entity.

-relation2id.txt:the first column is relation name, and second column is index of relation.

-train.txt , test.txt , valid.txt: the first column is index of subject entity, second column is index of relation, third column is index of object entity, fourth column is the start time of fact and fifth column is end time of fact.

-stat.txt: num of entites and num of relations

## Baselines

We use following public codes for baseline experiments. 

| Baselines   | Code                                                                      | Embedding size | Batch num |
|-------------|---------------------------------------------------------------------------|----------------|------------|
| TransE ([Bordes et al., 2013](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data))      | [Link](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/openke) | 100, 200       | 100, 200       |
| TransH ([Wang et al., 2014](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531/8546))   | [Link](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/openke) | 100, 200       | 100, 200      |
| t-TransE ([Leblay et al., 2018](https://dl.acm.org/doi/fullHtml/10.1145/3184558.3191639))    | [Link](https://github.com/INK-USC/RE-Net/tree/master/baselines)                                  | 50, 100, 200   | 100, 200       |
| TA-TransE ([Alberto et al., 2018](https://www.aclweb.org/anthology/D18-1516.pdf))      | [Link](https://github.com/INK-USC/RE-Net/tree/master/baselines)     | 100, 200            | Default    |
| HyTE ([Dasgupta et al., 2018](http://talukdar.net/papers/emnlp2018_HyTE.pdf))        | [Link](https://github.com/malllabiisc/HyTE)                               | Default            | Default    |

## Contact

For any questions or suggestions you can use the issues section or reach us at amaia.salvador@upc.edu or nhynes@mit.edu.
