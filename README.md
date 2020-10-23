# MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation
This repository provides a PyTorch implementation of the paper [MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://arxiv.org/pdf/1903.01945.pdf).

An extended version has been published in TPAMI [Link](https://github.com/sj-li/MS-TCN2).

Tested with:
- PyTorch 0.4.1
- Python 2.7.12


### Qualitative Results:

<div align="center">
  <a href="https://www.youtube.com/watch?v=9XphWB9w7p8"><img src="https://img.youtube.com/vi/9XphWB9w7p8/0.jpg" alt="IMAGE ALT TEXT"></a>
</div>

### Training:

* Download the [data](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) folder, which contains the features and the ground truth labels. (~30GB) (If you cannot download the data from the previous link, try to download it from [here](https://zenodo.org/record/3625992#.Xiv9jGhKhPY))
* Extract it so that you have the `data` folder in the same directory as `main.py`.
* To train the model run `python main.py --action=train --dataset=DS --split=SP` where `DS` is `breakfast`, `50salads` or `gtea`, and `SP` is the split number (1-5) for 50salads and (1-4) for the other datasets.

### Prediction:

Run `python main.py --action=predict --dataset=DS --split=SP`. 

### Evaluation:

Run `python eval.py --dataset=DS --split=SP`. 

### Citation:

If you use the code, please cite

    Y. Abu Farha and J. Gall.
    MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019

    S. Li, Y. Abu Farha, Y. Liu, MM. Cheng,  and J. Gall.
    MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation.
    In IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020
