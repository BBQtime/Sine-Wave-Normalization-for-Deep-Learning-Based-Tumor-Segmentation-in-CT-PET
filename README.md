# Sine-Wave-Transformations-for-Deep-Learning-Based-Tumor-Segmentation-in-CT-PET-Imaging
This repository contains the implementation of a deep learning model for automated tumor lesion segmentation in CT/PET scans, designed for the autoPET III Challenge. It introduces a novel SineNormalBlock, leveraging sine wave transformations to enhance PET data processing for improved segmentation accuracy.

The shared model are public available at : 
[Sine-Wave Normalization for Deep Learning Based Tumor Segmentation in CT-PET](https://huggingface.co/JintaoRen/Sine-Wave-Normalization-for-Deep-Learning-Based-Tumor-Segmentation-in-CT-PET)

### The main contribution of this study
SineNormalBlock:
```
nets
├── resnet.py
└── resnet_sin_normal.py
```

Trainer:
```
nnUNetTrainer
├── nnUNetTrainerResenc.py
├── nnUNetTrainerUmambaSinNorm.py
```
