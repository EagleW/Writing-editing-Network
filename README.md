# Writing-editing Network: Paper Abstract Writing through Editing Mechanism

[Paper Abstract Writing through Editing Mechanism](https://arxiv.org/pdf/1805.06064.pdf)

Accpeted by ACL 2018

Table of Contents
=================
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Citation](#citation)
  
## Requirements

#### Environment:

- [Pytorch 0.4](http://pytorch.org/)
-  Python 3.6

#### Data: 

- [ACL_titles_abstracts_dataset](https://github.com/EagleW/ACL_titles_abstracts_dataset)

## Quickstart

#### Preprocessing:
Put the acl_titles_and_abstracts.txt under the Writing-editing network folder. Randomly split the data into train, dev and test by runing split_data.py. 

#### Training
Hyperparameter can be adjust in the Config class of main.py.
```
python main.py --cuda --mode 0
```

#### Validation
Compute score:
```
python main.py --cuda --mode 3
```
Predict sentence:
```
python main.py --cuda --mode 1
```

## Citation
```
@InProceedings{Writ_edit,
  author ="Wang, Qingyun
            and Zhou, Zhihao
            and Huang, Lifu
            and Whitehead, Spencer
            and Zhang, Boliang
            and Ji, Heng
            and Knight, Kevin",
  title = 	"Paper Abstract Writing through Editing Mechanism",
  booktitle = 	"Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics",
  year = 	"2018"
}
```
