# MSP

A PyTorch implementation of _[TCI 2023 paper](https://ieeexplore.ieee.org/document/10034834), "Multi-Stream Progressive Restoration for Low-Light Light Field Enhancement and Denoising"_

> This repo is _NOT_ the official implementation of the paper. For the official implementation, visit [here](https://github.com/shuozh/MSP).

## Quick Start

First, download the [L3F dataset](https://mohitlamba94.github.io/L3Fnet/) and symlink it to the project root. The project directory structure should resemble:

```
|-- L3FNet
    |-- L3F-dataset
        |-- jpeg
            |-- train
            |-- test
    |-- train.py
    |-- eval.py
    ...
```

Modify `config.toml` to do necessary setup:


Modify the `split` item in `config.toml` to specify the subset for training/evaluation/testing:

``````
[env]
wandb_key = ''

[data]
split = '20'
``````

- `split`: specify the subset for training/evaluation/testing. Should be '20', '50' or '100'.
- `wandb_key`: (optional) the api key for wandb.

To start training or testing, execute the following commands:

```sh
# training
python train.py config.toml

# testing
python eval.py config.toml --ckpt ${CKPT_PATH}
```