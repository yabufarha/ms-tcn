# MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation
This repository provides a PyTorch implementation of the paper [MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation]().

Tested with:
- PyTorch 0.4.1
- Python 2.7.12

### Training:

* download the [data](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8) folder, which contains the features and the ground truth labels.
* extract it so that you have the `data` folder in the same directory as `main.py`.
* To train the model run `python main.py --action=train --dataset=DS --split=SP` where `DS` is `breakfast`, `50salads` or `gtea`, and `SP` is the split number (1-5) for 50salads and (1-4) for the other datasets.

### Prediction:

* Run `python main.py --action=predict --dataset=DS --split=SP` for evaluating the the model on split1 of Breakfast. 

### Evaluation:

Run `python eval.py ---dataset=DS --split=SP`. 

### Citation:

If you use the code, please cite

    Y. Abu Farha and J. Gall:
    MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation
    in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019
