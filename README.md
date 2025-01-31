# GEB+: A Benchmark for Generic Event Boundary Captioning, Grounding and Retrieval
New Official Implementations for [GEB+: A Benchmark for Generic Event Boundary Captioning, Grounding and Retrieval](https://arxiv.org/abs/2204.00486) [[PDF](https://arxiv.org/pdf/2204.00486.pdf)], ECCV 2022

[Yuxuan Wang](https://yuxuanw.me), [Difei Gao](https://scholar.google.com/citations?user=No9OsocAAAAJ&hl=en), [Licheng Yu](https://lichengunc.github.io), Stan Weixian Lei, Matt Feiszli, and [Mike Zheng Shou](https://sites.google.com/view/showlab)

**Updated in Jan. 2025:** 
For convenience, we provide downloads of **video frames** and our boundaries. 
The dataset is now easier to use, and we hope that this work will continue to contribute to the field, even as time goes by.

![image](https://github.com/Yuxuan-W/GEB-Plus/blob/main/figures/Cover.png)

We introduce a new dataset called **Kinetics-GEB+**. The dataset consists of over 170k boundaries associated with captions describing status changes in the generic events in 12K videos. Upon this new dataset, we propose three tasks (**Boundary Captioning**, **Boundary Grounding** and **Boundary Caption-Video Retrieval**) supporting the development of a more fine-grained, robust, and human-like understanding of videos through status changes.

<br/>

# Easily Using Kinetic-GEB+ Dataset:

## Download the Video Frames (Newly Released!)

**For ease of use, we provide our collection of video frames for direct download: [[Download](https://entuedu-my.sharepoint.com/:u:/g/personal/yuxuan003_e_ntu_edu_sg/EQBGnIiM3PpGs8VxfptwbIgBYnur9vuSpThH6qxUaQOU7w?e=DBAdkX)]!!**

The frame rate varies, reaching up to 30 FPS, depending on the source videos. 

Please note that the original videos are from the **Kinetics-400** Dataset. We only provide a mirrored version for convenience.
All copyrights remain with the **talented artists on YouTube who created these videos**.

## Directly Using Kinetics-GEB+ Dataset (Updated!)

**Here we release 2 versions of boundaries:**

a) **Filtered boundaries (Recommended) [[Download](https://entuedu-my.sharepoint.com/:f:/g/personal/yuxuan003_e_ntu_edu_sg/EgjogzMe_H5KquTEF7YaG0ABCpIF6rGFbaSSl1FVntfE9g?e=1wUtBt)]**  used in our paper, which has been adjusted for downstream task, including 40K filtered boundaries.

b) **Raw annotation [[Download](https://entuedu-my.sharepoint.com/:f:/g/personal/yuxuan003_e_ntu_edu_sg/EpLQLIbuoIVLhw3jHoEtyxsBoDt3sJfNr2m6LQyAPahsqg?e=C3ACFz)]**  that could be used as supplement in training your own model, including 170K boundaries.

Note that our paper uses version a) in the evaluation.

**What is the difference**: In our **Kinetics-GEB+** dataset, each video contains 1 to 8 annotations from different annotators and each annotation consists of several boundaries inside a video, where the locations are not the same. 
In the filtered version, we select one annotator whose labeled boundaries are most consistent with others to reduce noise and duplication. 
Then, we use these boundaries’ timestamps as the anchors to merge other annotators’ captions, preserving the diversity of different opinions, which results in 40k anchor boundaries from all videos.


<br/>

# Using Our Baseline Models

**Updated in Jan. 2025:** Due to the fast development of video understanding field, these baselines may be out of date.

**It is more suggested to just using our provided annotations and video frames.**

<details>
  <summary>Click to expand</summary>

## Installation

We evaluate many representative baselines in our dataset, where we also design a new **TPD (Temporal-based Pairwise Difference) Modeling** method for visual difference and achieve significant performance improvements. Besides, the results show there are still formidable challenges for current methods in the utilization of different granularities, representation of visual difference, and the accurate localization of status changes. Further analysis shows that our dataset can drive developing more powerful methods to understand status changes and thus improve video level comprehension.

![image](https://github.com/Yuxuan-W/GEB-Plus/blob/main/figures/Tasks.png)

Clone the project to run our baseline models:

`git clone https://github.com/Yuxuan-W/GEB-Plus.git`

Clone our conda environment using:

`conda env create -n ENVNAME --file environment.yml`

Note that the version of `pytorch-transformer` we use is `1.0.0`.

<br/>

## Task1: Boundary Captioning
![image](https://github.com/Yuxuan-W/GEB-Plus/blob/main/figures/Captioning_res.png)
### Preparing evaluation package
To run Boundary Captioning task, you need to download the **evaluation package** [[Download](https://github.com/LuoweiZhou/coco-caption/tree/de6f385503ac9a4305a1dcdc39c02312f9fa13fc/pycocoevalcap)] and put it under `utils` folder as:

`GEBC/utils/pycocoevalcap`

Note that the evaluation package also requires **Java**, one simple way is to install a light-weight open-jdk on your server if you haven't installed.

### Preparing features
To run Boundary Captioning task, you need to download and unzip the **features** [[Download](https://entuedu-my.sharepoint.com/:f:/g/personal/yuxuan003_e_ntu_edu_sg/EmK42CKOoHpBl28pkglYf4cB3-UuEWm_1JThKwYnLuNWcw?e=PYt1Os)], make sure you have the following path:

`GEBC/datasets/features/region_feature`

`GEBC/datasets/features/tsn_captioning_feature`

### Training from scratch
To train on the captioning baseline, execute the following command:

`python run_captioning.py --do_train --do_test --do_eval --ablation obj --evaluate_during_training`

### Testing our trained model
We only provide the checkpoint that generating our highest score in the paper [[Download](https://entuedu-my.sharepoint.com/:f:/g/personal/yuxuan003_e_ntu_edu_sg/EgAn4YTeqf5OqcFCBryU3IABhlmTiz9_y1TY91VuJLIOmQ?e=9GRM0J)].
Unzip the folder to your project, execute the following command:

`python run_captioning.py --do_test --do_eval --ablation obj --eval_model_dir $YOUR_UNZIPPED_DIR$`

### Performance of our baseline
The best performance of our baseline are achieved by _ActBERT-revised_ with _ResNet-roi+TSN_ feature:

|        | _Subject_  | _Status Before_ | _Status After_ | **Average** |
| :----: | :----: | :----: | :----: | :----: |
|  _CIDEr_ | 85.33  | 75.98 | 62.82 | **74.71** | 
| _SPICE_  | 20.10  | 20.66 | 17.81 | **19.52** | 
| _ROUGE_L_  | 39.16  | 23.70 | 21.60 | **28.15** | 


<br/>

## Task2: Boundary Grounding
![image](https://github.com/Yuxuan-W/GEB-Plus/blob/main/figures/Grounding_res.png)

Like we mentioned in the paper, we use two schemes of frame sampling when proposing the timestamp candidates who might be the answer. By default, we sampled one candidates every 3 frames (0.1s), or we used the baseline of GEBD to generate proposals. Here we provide implementations for both of them.

### Preparing features
To run Boundary Grounding task, you need to download and unzip the **features** [[Download](https://entuedu-my.sharepoint.com/:f:/g/personal/yuxuan003_e_ntu_edu_sg/EmK42CKOoHpBl28pkglYf4cB3-UuEWm_1JThKwYnLuNWcw?e=PYt1Os)], make sure you have the following path:

`GEBC/datasets/features/region_feature`

If not using GEBD proposals (By Default), you will need:
`GEBC/datasets/features/tsn_all1s_feature`

If using GEBD proposals, you will need:
`GEBC/datasets/features/tsn_gebd_feature`

### Training from scratch
To train on the grounding baseline, execute the following command:

`python run_grounding.py --do_train --do_test --do_eval --ablation obj --evaluate_during_training` (By Default)

Or if you want to use GEBD proposals in followed validation and testing after training is finished:

`python run_grounding.py --do_train --do_test --do_eval --use_gebd --ablation obj --evaluate_during_training` (Use GEBD)

### Testing our trained model
We only provide the checkpoint that generating our highest score in the paper [[Download](https://entuedu-my.sharepoint.com/:f:/g/personal/yuxuan003_e_ntu_edu_sg/EgAn4YTeqf5OqcFCBryU3IABhlmTiz9_y1TY91VuJLIOmQ?e=9GRM0J)].
Unzip the folder to your project, execute the following command:

`python run_grounding.py --do_test --do_eval --ablation obj --eval_model_dir $YOUR_UNZIPPED_DIR$` (By Default)

Or if you want to use GEBD proposals in testing:

`python run_grounding.py --do_test --do_eval --use_gebd --ablation obj --eval_model_dir $YOUR_UNZIPPED_DIR$` (Use GEBD)

### Performance of our baseline
The best performance of our baseline are achieved by _FROZEN-revised_ with _ResNet-roi+TSN_ feature:

| Threshold(s) | 0.1 | 0.2 | 0.3 | 1 | 1.5 | 2 | 2.5 | 3 | **Average** |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| **_Default_** | 4.28 | 8.54	| 18.33 |	31.04 |	40.48	| 47.86	| 54.81	| 61.45 | **33.35** |
| _Use GEBD_ | 4.20 | 8.48	| 18.49	| 29.91	| 39.54	| 48.37	| 55.29	| 61.55 | **33.32** |

<br/>

## Task3: Boundary Caption-Text Retrieval
![image](https://github.com/Yuxuan-W/GEB-Plus/blob/main/figures/Retrieval_res.png)

### Preparing features
To run Boundary Grounding task, you need to download and unzip the **features** [[Download](https://entuedu-my.sharepoint.com/:f:/g/personal/yuxuan003_e_ntu_edu_sg/EmK42CKOoHpBl28pkglYf4cB3-UuEWm_1JThKwYnLuNWcw?e=PYt1Os)], make sure you have the following path:

`GEBC/datasets/features/region_feature`

`GEBC/datasets/features/tsn_gebd_feature`

### Training from scratch
To train on the grounding baseline, execute the following command:

`python run_retrieval.py --do_train --do_test --do_eval --ablation obj --evaluate_during_training`

### Testing our trained model
We only provide the checkpoint that generating our highest score in the paper [[Download](https://entuedu-my.sharepoint.com/:f:/g/personal/yuxuan003_e_ntu_edu_sg/EgAn4YTeqf5OqcFCBryU3IABhlmTiz9_y1TY91VuJLIOmQ?e=9GRM0J)].
Unzip the folder to your project, execute the following command:

`python run_retrieval.py --do_test --do_eval --ablation obj --eval_model_dir $YOUR_UNZIPPED_DIR$`

### Performance of our baseline
The best performance of our baseline are achieved by _FROZEN-revised_ with _ResNet-roi+TSN_ feature extracted following the timestamps proposals generated by GEBD baseline:

| Metric | mAP | R@1 | R@5 | R@10 | R@50 |
| :----: | :----: | :----: | :----: | :----: | :----: |
| _Use GEBD_ | 23.39 |	12.80	| 34.81	| 45.66 |	68.1 |

</details>

<br/>

## Citation

If you find our work helps, please cite our paper:
```
@article{wang2022generic,
  title={Generic Event Boundary Captioning: A Benchmark for Status Changes Understanding},
  author={Wang, Yuxuan and Gao, Difei and Yu, Licheng and Lei, Stan Weixian and Feiszli, Matt and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2204.00486},
  year={2022}
}
```

<br/>

## Acknowledgement

Thanks to [Difei Gao](https://scholar.google.com/citations?user=No9OsocAAAAJ&hl=en), [Licheng Yu](https://lichengunc.github.io), and the great efforts contributed by other excellent staffs from [Meta AI](https://ai.facebook.com).

Thanks the amazing YouTube artists for creating these videos and the contributors of the original Kinetic-400 dataset.

This project is supported by the National Research Foundation, Singapore under its NRFF Award NRF-NRFF13-2021-0008, and [Mike Zheng Shou](https://sites.google.com/view/showlab)'s Start-Up Grant from NUS. The computational work for this article was partially performed on resources of the National Supercomputing Centre, Singapore.
