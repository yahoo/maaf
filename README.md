# Modality-Agnostic Attention Fusion for visual search with text feedback

This repository was used for the experiments described in [our paper](https://arxiv.org/abs/2007.00145), which can be cited as:
```
@article{dodds2020modality,
  title={Modality-Agnostic Attention Fusion for visual search with text feedback},
  author={Dodds, Eric and Culpepper, Jack and Herdade, Simao and Zhang, Yang and Boakye, Kofi},
  journal={arXiv preprint arXiv:2007.00145},
  year={2020}
}
```

The core model and training code is based on [TIRG code](https://github.com/google/tirg) written by the authors of [Composing Text and Image for Image Retrieval - An Empirical Odyssey](https://arxiv.org/abs/1812.07119), and adapted for the Fashion-IQ dataset partly using instructions [here](https://github.com/lugiavn/notes/blob/master/fashioniq_tirg.md). The included Transformer code is adapted from [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html). 
Further modifications are our own.

## Setup

The code is tested on Python 3.6 with PyTorch 1.5 and should also work on newer versions. The following can be installed with pip:

- pytorch
- torchvision
- numpy
- tqdm
- tensorboardX
- pillow
- pytorch_pretrained_bert
- gitpython

## Datasets

We do not own any of the datasets used in our experiments here. Below we link to the datasets where we acquired them.

### Fashion IQ

Download the dataset from [here](https://github.com/XiaoxiaoGuo/fashion-iq).

### Birds-to-Words

Download the dataset from [here](https://github.com/google-research-datasets/birds-to-words). Our data loader handles reinterpreting this dataset for text-guided image retrieval.

### Spot-the-Diff

Download the dataset from [here](https://github.com/harsh19/spot-the-diff). Our data loader handles reinterpreting this dataset for text-guided image retrieval. Note that the train/val/test splits are not designed for this task, and metrics may differ substantially between evaluations on val versus test although each is individually meaningful.

### CSS3D dataset

Download the dataset from this [external website](https://drive.google.com/file/d/1wPqMw-HKmXUG2qTgYBiTNUnjz83hA2tY/view?usp=sharing).

Make sure the dataset includes these files:
`<dataset_path>/css_toy_dataset_novel2_small.dup.npy`
`<dataset_path>/images/*.png`

### MITStates dataset
Download the dataset from this [external website](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html).

Make sure the dataset include these files:

`<dataset_path>/images/<adj noun>/*.jpg`


### Fashion200k dataset
Download the dataset from this [external website](https://github.com/xthan/fashion-200k) Download our generated test_queries.txt from [here](https://storage.googleapis.com/image_retrieval_css/test_queries.txt).

Make sure the dataset include these files:

```
<dataset_path>/labels/*.txt
<dataset_path>/women/<category>/<caption>/<id>/*.jpeg
<dataset_path>/test_queries.txt`
```

## Notes

Pytorch's data loader might consume a lot of memory, if that's an issue add `--loader_num_workers=0` to disable loading data in parallel.
