# Writing-editing Network: Paper Abstract Writing through Editing Mechanism

[Paper Abstract Writing through Editing Mechanism](http://aclanthology.org/P18-2042.pdf)

[[Poster]](https://eaglew.github.io/files/Paper_abstract_generation.pdf)[[Fake Handbook*]](https://eaglew.github.io/files/handbook.pdf) *Fake abstracts for the main conference (ACL 2018)

Accpeted by 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018)


Table of Contents
=================
  * [Overview](#overview)
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Citation](#citation)

## Overview
<p align="center">
  <img src="https://eaglew.github.io/images/writing-editing.png?raw=true" alt="Photo" style="width="100%;"/>
</p>

## Requirements

#### Environment:

- [Pytorch 0.4](http://pytorch.org/)
-  Python 3.6 **CAUTION!! Model will not be saved and loaded properly under Python 3.5**

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
Predict abstract:
```
python main.py --cuda --mode 1
```

## Citation
```
@InProceedings{P18-2042,
  author = 	"Wang, Qingyun
		and Zhou, Zhihao
		and Huang, Lifu
		and Whitehead, Spencer
		and Zhang, Boliang
		and Ji, Heng
		and Knight, Kevin",
  title = 	"Paper Abstract Writing through Editing Mechanism",
  booktitle = 	"Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"260--265",
  location = 	"Melbourne, Australia",
  url = 	"http://aclanthology.org/P18-2042"
}
```
