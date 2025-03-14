# Multimodal-Action-Recognition
This repo is the official implementation for **EPAM-Net: An Efficient Pose-driven Attention-guided Multimodal Network for Video Action Recognition**, which is accepted by **Neurocomputing 2025**.
[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://www.arxiv.org/abs/2408.05421) <br />
Paper: https://www.sciencedirect.com/science/article/pii/S0925231225004539

## Introduction
Existing multimodal-based human action recognition approaches are either computationally expensive, which limits their applicability in real-time scenarios, or fail to exploit the spatial temporal information of multiple data modalities. In this work, we present a novel and efficient pose-driven attention-guided multimodal network (EPAM-Net) for action recognition in videos. Specifically, we propose eXpand temporal Shift (X-ShiftNet) convolutional architectures for RGB and pose streams to capture spatio-temporal features from RGB videos and their skeleton sequences. The X-ShiftNet tackles the high computational cost of the 3D CNNs by integrating the Temporal Shift Module (TSM) into an efficient 2D CNN, enabling efficient spatiotemporal learning. Then skeleton features are utilized to guide the visual network stream, focusing on keyframes and their salient spatial regions using the proposed spatial–temporal attention block. Finally, the predictions of the two streams are fused for final classification.. The proposed architecture achieved comparative performance with state-of-the-art methods on NTU RGB-D 60, NTU RGB-D 120, PKU-MMD, and Toyota SmartHome datasets with up to a 72.8x reduction in FLOPs and up to a 48.6x reduction in the number of network parameters.


# Download dataset
1. **NTU-RGB+D 60** dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
2. **NTU-RGB+D 120** dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
3. **PKU-MMD** dataset from [https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html)
4. **Toyota SmartHome** trimmed dataset from [https://project.inria.fr/toyotasmarthome/](https://project.inria.fr/toyotasmarthome/)

# Architecture of EPAM-Net
<div align=center>
<img src ="./figures/EMAP-Net Architecture.png" width="1000"/>
</div>
 
# Result
We report the mean Top-1 accuracy (%) for Toyota-Smarthome dataset and Top-1 accuracy (%) for other datasets using 1-clip per video.
| Method | NTU-60 X-Sub | NTU-60 X-View | NTU-120 X-Sub | NTU-120 X-Set | Toyota Smarthome X-Sub |Toyota Smarthome X-View2| 
| ------ | ------------ | ------------- | ------------- | ------------- | -------
|  EPAM-Net  |     96.1%    |      99.0%    |      92.4%    |      94.3% |  71.7%  |67.8% |

# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- We provide the dependency file of our experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running `pip install -r requirements.txt `

# Data Preparation

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

# Citation
Please cite this work if you find it useful:
```BibTex
@article{abdelkawy2025epam,
  title={EPAM-Net: An efficient pose-driven attention-guided multimodal network for video action recognition},
  author={Abdelkawy, Ahmed and Ali, Asem and Farag, Aly},
  journal={Neurocomputing},
  pages={129781},
  year={2025},
  publisher={Elsevier}
}
```
