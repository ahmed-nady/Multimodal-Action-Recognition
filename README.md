# Multimodal-Action-Recognition
In our work entitled "EPAM-Net: An Efficient Pose-driven Attention-guided Multimodal Network for Video Action Recognition," we address the limitations of existing multimodal-based human action recognition approaches that are either computationally expensive, which limits their applicability in real-time scenarios, or fail to exploit the spatial-temporal information of multiple data modalities. 

We present an efficient multimodal architecture (EPAM-Net) with a spatial temporal attention block that utilizes skeleton features to help the visual network stream focus on key frames and their salient spatial regions. The proposed architecture achieved comparative performance with state-of-the-art methods on NTU RGB-D 60, NTU RGB-D 120, PKU-MMD, and Toyota
SmartHome datasets with up to a 72.8x reduction in FLOPs and up to a 48.6x reduction in the number of network parameters.
 
Paper: https://lnkd.in/ec-hWUzq

## Results and Models

### NTU60

### NTU120

### PKU-MMD

### Toyota-Smarthome

## Train

## Inference 
You can use the following command to test a unimodal model.
```shell
python inference/test.py
```
You can use the following command to test a multimodal model.
```shell
python inference/test_multimodal.py
```
