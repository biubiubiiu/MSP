# MSP

A PyTorch implementation of _[TCI 2023 paper](https://ieeexplore.ieee.org/document/10034834), "Multi-Stream Progressive Restoration for Low-Light Light Field Enhancement and Denoising"_

**Important Note:** This repository is **not** the official implementation of the paper. For the official implementation, please refer to [this repository](https://github.com/shuozh/MSP).

## Quick Start

1. Download the [L3F dataset](https://mohitlamba94.github.io/L3Fnet/) and create a symlink to the project root. Ensure your directory structure resembles the following:
   
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

2. Modify `config.toml` for necessary setup:
   
   ```toml
   [env]
   wandb_key = ''
   
   [data]
   split = '20'
   cropped_resolution = 5
   ```
- `wandb_key`: (optional) API key for wandb.

- `split`: the subset for training/evaluation/testing ('20', '50', or '100').

- `cropped_resolution`: angular resolution of the cropped LF image.
3. To start training or testing, execute the following commands:
   
   ```sh
   # training
   python train.py config.toml
   
   # testing
   python eval.py config.toml --ckpt ${CKPT_PATH}
   ```

## Results

Using center $5 \times 5$ views for training and evaluation:

|            | LSF-20                                                                                                                                                                           | LSF-50                                                                                                                                                                           | LSF-100                                                                                                                                                                          |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Paper      | 25.48 / 0.84                                                                                                                                                                     | 24.10 / 0.78                                                                                                                                                                     | 22.75 / 0.73                                                                                                                                                                     |
| Reproduced | 25.27 / 0.83 [model](https://mega.nz/file/pIkR2AyS#TPQKH_tcAPB7HtBPEliX7yKu6uOtb31DrHsvYNoqAX8) [log](https://mega.nz/file/9c83wIpA#wPZYZmpT2RMufKw3UK3F3TwUpT1l2tS2-JyhMj_oY5E) | 23.58 / 0.78 [model](https://mega.nz/file/VYVUVRJY#tAWPZdQvKrnC-92X0e3nENZrrvJx145Cg__w_1KpZlA) [log](https://mega.nz/file/wVtikBKD#Vokb0aHTnvbnHzJaJVRGy7gDFd46OuI10aydsuKoH5Y) | 22.53 / 0.73 [model](https://mega.nz/file/kEV1iC6Q#OtJmhOK5qeGQ2LNJpHdIp-o6fGTRXMQIFBWub-QqqzM) [log](https://mega.nz/file/Jc1AXQwa#3-kLFQBjlmIxVbcSDr_kGbNpPZgc5HiLBbIVHYI5g4U) |

> NOTE: PSNR and SSIM are computed after quantizing the output to the uint8 data type, which differs from the official implementation.
