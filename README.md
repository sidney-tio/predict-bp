# Blood Pressure Measurement via PPG

## Overview
This repo contains the required modules to train a deeplearning-based BloodPressure
predictor from PPG readings.

The CNN-based model is the 1D version of ResNet, as recommended by [PulseDB paper](https://www.frontiersin.org/articles/10.3389/fdgth.2022.1090854/full). The original implementation of the ResNet1D can be found [here](https://github.com/hsd1503/resnet1d).

The RNN-based model is based on this [repo](https://github.com/psu1/DeepRNN/tree/master).

## Requirements
This repo uses Pytorch and Pytorch lightning for its training modules. This repo was trained using Python 3.10.10.

## Dataset
The dataset used to train the models are taken from the [PulseDB Vital dataset](https://github.com/pulselabteam/PulseDB). The Vitals dataset is selected as it contains Height, Weight, and Age records for comparison with the heuristic method provided by Bonfire.AI. It could be used downstream as features to enhance the predictive power of the deep learning models.

The files provided by the original dataset are in `.mat` format. It is recommended to use `h5py` or your own MATLAB method to parse the data.
For this particular repo, we only provide the code to load the parsed data from `.npy` file formats.


The original dataset samples PPG at 125hz for 10s; as such, each sample contains 1250 PPG readings to predict the sysolic blood pressure (SBP) and diastolic blood pressure (DBP). Both SBP and DBP are a single scalar for each reading.

## Training
After installing the required packages from `requirements.txt`,

```
python train.py
```
or
```
python train_deeprnn.py
```

If you wish to perform hyperparameter tuning, you may refer to `config.yml` or `config_deeprnn.yml` for the relevant parameters. It is also recommended to read Pytorch Lightnings documentation on the possible QoL enhancements to the training process.

## Results
The respective algorithms achieved the following metrics on a specific test set:

|                   | SBP MAE | DBP MAE | Overall MAE | SBP RMSE | DBP RMSE | Overall RMSE |
|-------------------|---------|---------|-------------|----------|----------|--------------|
| ResNet1D          | 14.10  | 8.76   | 11.55      | 18.11   | 10.89   | 15.36       |
| Bonfire Heuristic | 42.646  | 25.817  | 34.732      | 62.499   | 30.612   | 46.556       |
| DeepRNN           | 14.32     | 9.26     | 11.80         | 18.32      | 11.65      | 15.50          |


## Extension

Age and other meta data could be added into the enhance the ResNet1D model for more accurate figures.
