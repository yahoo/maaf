# Modality-Agnostic Attention Fusion for visual search with text feedback

The methods in this repository were used for the experiments described in two papers from Yahoo's Visual Intelligence Research team. The more recent paper's main results can be reproduced using the scripts in the `experiment_scripts` directory. If you find this code useful please cite
```
@article{dodds2022training,
  title = {Training and challenging models for text-guided fashion image retrieval},
  author = {Dodds, Eric and Culpepper, Jack and Srivastava, Gaurav},
  journal={arXiv preprint arXiv:2204.11004}
  year = {2022},
  doi = {10.48550/ARXIV.2204.11004},
}
```

We also recommend using the latest version of the code if you wish to build upon our general methods. However if you are interested specifically in reproducing the results in our earlier paper or using datasets discussed there, it will likely be easier to start from commit [49a0df9](https://github.com/yahoo/maaf/commit/49a0df90baf4b9d4a194ed646620375b5b837b15). The [earlier paper](https://arxiv.org/abs/2007.00145) can be cited as:
```
@article{dodds2020modality,
  title={Modality-Agnostic Attention Fusion for visual search with text feedback},
  author={Dodds, Eric and Culpepper, Jack and Herdade, Simao and Zhang, Yang and Boakye, Kofi},
  journal={arXiv preprint arXiv:2007.00145},
  year={2020}
}
```

This codebase was originally adapted from [TIRG code](https://github.com/google/tirg) written by the authors of [Composing Text and Image for Image Retrieval - An Empirical Odyssey](https://arxiv.org/abs/1812.07119). The core model and training code is based on. Transformer code is adapted from [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html). Further modifications are our own.
We use [YACS](https://github.com/rbgirshick/yacs) for configurations.

## Setup

The code is tested on Python 3.6 with PyTorch 1.5 and should also work on newer versions. Installing with pip should install the requirements.

## Datasets

### Challenging Fashion Queries (CFQ)

The Challenging Fashion Queries dataset described in our paper can be found [here](https://webscope.sandbox.yahoo.com/catalog.php?datatype=a&did=92) and used for research purposes.

We do not own any of other datasets used in our experiments here. Below we link to the datasets where we acquired them.

### Fashion IQ

Download the dataset from [here](https://github.com/XiaoxiaoGuo/fashion-iq).
