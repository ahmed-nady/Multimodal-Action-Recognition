## Multimodal-Action-Recognition
This repo is the official implementation for **EPAM-Net: An Efficient Pose-driven Attention-guided Multimodal Network for Video Action Recognition**, which is accepted by **Neurocomputing 2025**.
[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://www.arxiv.org/abs/2408.05421) <br />
Paper: https://www.sciencedirect.com/science/article/pii/S0925231225004539

## Abstract
Existing multimodal-based human action recognition approaches are either computationally expensive, which limits their applicability in real-time scenarios, or fail to exploit the spatial temporal information of multiple data modalities. In this work, we present a novel and efficient pose-driven attention-guided multimodal network (EPAM-Net) for action recognition in videos. Specifically, we propose eXpand temporal Shift (X-ShiftNet) convolutional architectures for RGB and pose streams to capture spatio-temporal features from RGB videos and their skeleton sequences. The X-ShiftNet tackles the high computational cost of the 3D CNNs by integrating the Temporal Shift Module (TSM) into an efficient 2D CNN, enabling efficient spatiotemporal learning. Then skeleton features are utilized to guide the visual network stream, focusing on keyframes and their salient spatial regions using the proposed spatial–temporal attention block. Finally, the predictions of the two streams are fused for final classification.. The proposed architecture achieved comparative performance with state-of-the-art methods on NTU RGB-D 60, NTU RGB-D 120, PKU-MMD, and Toyota SmartHome datasets with up to a 72.8x reduction in FLOPs and up to a 48.6x reduction in the number of network parameters.

# Architecture of EPAM-Net
<div align=center>
<img src ="./figures/EMAP-Net Architecture.png" width="1000"/>
</div>

# Prerequisites
- Python3
- [PyTorch](http://pytorch.org/)
- We provide the dependency file of our experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running `pip install -r requirements.txt `

# Download dataset
1. **NTU-RGB+D 60** dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
2. **NTU-RGB+D 120** dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
3. **PKU-MMD** dataset from [https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html)
4. **Toyota SmartHome** trimmed dataset from [https://project.inria.fr/toyotasmarthome/](https://project.inria.fr/toyotasmarthome/)

# Data Preparation
**For Pose estimation**, we utilize a Top-Down pose estimation approach instantiated with HRNet-W32 to extract 2D poses
from videos and save the coordinate triplets (x, y, score) following [PoseConv work](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md) 
Here, links to the pre-processed skeleton annotations, you can directly download them and use them for training & testing.
For NTU 60 and 120 dataset, you can use the script file data_preparation/prepare_ntu_dataset_annotations.py to split such pose annotations (ntu120_2d.pkl) into X-Sub, X-Set for NTU120 and similarly for NTU60.
- NTURGB+D \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl
- NTURGB+D 120 [2D Skeleton]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu120_2d.pkl
- PKU [2D Skeleton]: https://drive.google.com/drive/folders/10LIsMsJIWuo3g3FfywT3f6HEy--db4T0?usp=drive_link
- Toyota Smarthome [2D Skelton]: https://drive.google.com/drive/folders/1DtHneH1oH6YPJ2TdB3tfKcJcl2bTDb3d?usp=drive_link

**For RGB videos**, We crop the RGB frames with the global box, which envelops all persons in the video, and resize them to a resolution of 224
x 224 using script **data_preparation/crop_RGBFrames_PKU.py** (PKU dataset is an example). The cropped RGB frames are used as input for the RGB stream.

For PKU, since it is an untrimmed dataset, we extract the actions clips from each video using the script **data_preparation/pku_dataset_videos_split.py**. 

# Testing Pretrained Models
You may download the trained models reported in the paper via [GoogleDrive](https://drive.google.com/drive/folders/1I0To7YHzSlbjpypomjkcE_DnWK01knB1?usp=drive_link) and put them in folder pretrained_EPAM_models.

# Evaluation 

You can use the following command to test a multimodal EPAM-Net on NTU 60, NTU 120, PKU and Toyota Smarthome datasets.
```shell
python inference/test_multimodal.py --dataset 'toyota' --evaluation_protocol 'xsub'
```
# Result
We report the mean Top-1 accuracy (%) for Toyota-Smarthome dataset and Top-1 accuracy (%) for other datasets using 1-clip per video.
| Method | NTU-60 X-Sub | NTU-60 X-View | NTU-120 X-Sub | NTU-120 X-Set | Toyota Smarthome X-Sub |Toyota Smarthome X-View2| 
| ------ | ------------ | ------------- | ------------- | ------------- |  ------------- |------------- |
|  EPAM-Net  |     96.1%    |      99.0%    |      92.4%    |      94.3% |  71.7%  |67.8% |

## Train
To train a new EPAM-Net, you need to train submodels for two inputs: skeleton joint, and RGB video, or you can download the pretrained submodels via [GoogleDrive](https://drive.google.com/drive/folders/1b4bUxybR4X8gvv9TJszYpiFOJNVdicdG?usp=sharing). After that, you can use the following command to train a unimodal model.
```shell
python train/ddp_train_MMActionRecognition_model_pytorch.py --dataset 'pku' --evaluation_protocol 'xsub'
```
## Acknowledgements
This repo is based on [MMAction2]([https://github.com/Uason-Chen/CTR-GCN](https://github.com/open-mmlab/mmaction2)).
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
