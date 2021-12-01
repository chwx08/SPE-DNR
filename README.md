# Spherical-Patches-Extraction and Deep-Learning Based Neuron Reconstructor (SPE-DNR) [[paper]](https://doi.org/10.1109/TMI.2021.3130934)

## Introduction
This is the source code for the spherical-patches-extraction and deep-learning based neuron reconstructor (SPE-DNR) with pretrained weights.   
SPE-DNR is an automatic method for neuron reconstruction from 3D microscopy images.

## Requirments
* pytorch==1.7.1  
* numpy==1.19.2  
* scipy==1.5.2  
* scikit-image==0.17.2  
* libtiff  
* tqdm  

## Usage
python main.py

## Pretrained Weights
In folder './checkpoint/classification_checkpoints'.

## Test Samples
One test image with corresponding seed maps and soma masks is in folder './test_samples'.

## Citation
If the code or method help with your research, please cite the following paper:
'''
@article{Chen2021deep,
  title={Deep-learning based automated neuron reconstruction from 3D microscopy images using synthetic training images},
  author={Chen, Weixun and Liu, Min and Du, Hao and Radojević, Miroslav and Wang, Yaonan and Meijering, Erik},
  journal={IEEE Transactions on Medical Imaging},
  year={2021},
  doi={https://doi.org/10.1109/TMI.2021.3130934}
}
'''
