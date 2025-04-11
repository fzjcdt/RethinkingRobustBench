# Rethinking RobustBench: Is High Synthetic-Test Data Similarity an Implicit Information Advantage Inflating Robustness Scores?

## Overview

This repository contains the code for the paper "Rethinking RobustBench: Is High Synthetic-Test Data Similarity an Implicit Information Advantage Inflating Robustness Scores?".

## Data

### Synthetic Data (20M)

To replicate the feature extraction and similarity calculation, you need the 20 million synthetic CIFAR-10 images used in the study. These can be downloaded from the links provided in the [DM-Improves-AT repository](https://github.com/wzekai99/DM-Improves-AT):

| Dataset   | Size | Download Links                                                                                                                                                            |
| :-------- | :--- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CIFAR-10  | 20M  | [part1](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/20m_part1.npz) [part2](https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/20m_part2.npz) |

Download both `20m_part1.npz` and `20m_part2.npz` and place them in your data directory.

### Pre-computed Top-20 Similarity Data (`20m_top20.npz`)

As an alternative to running the similarity calculation yourself, we provide the pre-computed file containing the top 20 most similar synthetic images (from the 20M set) for each CIFAR-10 test image, based on feature similarity.

**Download Links:**

*   **Google Drive:** [https://drive.google.com/file/d/1dPktooDpNOmnuiGaKQ8olBd-yFGkeIPO/view?usp=sharing](https://drive.google.com/file/d/1dPktooDpNOmnuiGaKQ8olBd-yFGkeIPO/view?usp=sharing)
*   **Baidu Netdisk:** [https://pan.baidu.com/s/1PYMsULBaF0I-aSPgCsIpZA?pwd=yx5u](https://pan.baidu.com/s/1PYMsULBaF0I-aSPgCsIpZA?pwd=yx5u) (Password: `yx5u`)

**File Contents:**

This `.npz` file contains NumPy arrays accessible via the following keys:

*   `image`: The actual pixel data of the top-k similar synthetic images. Shape: `(num_test_images * k, height, width, channels)`
*   `label`: The corresponding labels of these synthetic images. Shape: `(num_test_images * k,)`
*   `test_image_id`: The index (0-9999) of the CIFAR-10 test image to which each synthetic image is most similar. Shape: `(num_test_images * k,)`
*   `rank`: The similarity rank (0 to k-1, where 0 is the most similar). Shape: `(num_test_images * k,)`
*   `similarity`: The calculated cosine similarity score. Shape: `(num_test_images * k,)`

In our case, `k=20` and `num_test_images=10000`.

## Code Usage

The following scripts are provided to perform the feature extraction and similarity analysis:

1.  **`extract_features.py`**: Extracts deep features from the 20M synthetic dataset (`20m_part1.npz`, `20m_part2.npz`).
    *   *Usage:* Modify paths within the script and run `python extract_features.py`.
2.  **`extract_cifar_test_features.py`**: Extracts deep features from the standard CIFAR-10 test set.
    *   *Usage:* Modify paths within the script and run `python extract_cifar_test_features.py`.
3.  **`calc_similarity.py`**: Calculates the cosine similarity between the features extracted in the previous steps. It identifies the most similar synthetic images for each test image.
    *   *Usage:* Ensure the feature files from steps 1 & 2 are available. Modify paths within the script and run `python calc_similarity.py`.
4.  **`save_data.py`**: Saves the top-k (e.g., top 20) most similar synthetic images, their labels, corresponding test image IDs, ranks, and similarity scores into the `20m_top20.npz` format described above.
    *   *Usage:* Ensure the similarity results from step 3 are available. Modify paths and parameters (like `k=20`) within the script and run `python save_data.py`.

**Note:** You can skip running these scripts if you download the pre-computed `20m_top20.npz` file.

## Training Example (Reproducing Paper Results)

To train the WideResNet-28-10 model using the pre-computed `20m_top20.npz` data (as auxiliary data for TRADES), you can use the training script from the [DM-Improves-AT repository](https://github.com/wzekai99/DM-Improves-AT).

Run the following command, ensuring you replace `'./20m_top20.npz'` with the actual path to your downloaded or generated file:

```bash
python train-wa.py --data-dir 'dataset-data' \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10s_lr0p2_TRADES5_epoch400_bs512_fraction0p8_ls0p1' \
    --data cifar10s \
    --batch-size 512 \
    --model wrn-28-10-swish \
    --num-adv-epochs 400 \
    --lr 0.2 \
    --beta 5.0 \
    --unsup-fraction 0.8 \
    --aux-data-filename './20m_top20.npz' \
    --ls 0.1
```

## Pre-trained Models

Our pre-trained models are available for download from the following locations:

*   **Google Drive:** [https://drive.google.com/drive/folders/1UI7GDs0-EvqpEFOO1qEIIw44FhQk1gmR?usp=sharing](https://drive.google.com/drive/folders/1UI7GDs0-EvqpEFOO1qEIIw44FhQk1gmR?usp=sharing)
*   **Baidu Netdisk:** [https://pan.baidu.com/s/1T9G3ei44hslzRpcJM4kD5A?pwd=1s9q ](https://pan.baidu.com/s/1T9G3ei44hslzRpcJM4kD5A?pwd=1s9q) (Password: `1s9q`)

### Evaluation

To evaluate the downloaded models using AutoAttack, you can use the `eval-last-aa.py` script from the [DM-Improves-AT repository](https://github.com/wzekai99/DM-Improves-AT).

First, ensure you have placed the downloaded model files (e.g., the folder `WRN28-10Swish_cifar10s_lr0p2_TRADES5_epoch400_bs512_fraction0p8_ls0p1`) inside the directory specified by `--log-dir` (default is `trained_models`).

Then, run the following command:

```bash
python eval-last-aa.py --data-dir 'dataset-data' \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10s_lr0p2_TRADES5_epoch400_bs512_fraction0p8_ls0p1'
```
