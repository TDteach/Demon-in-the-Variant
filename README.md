# Demon in the Variant: Statistical Analysis of DNNs for Robust Backdoor Contamination Detection

### BEFORE YOU RUN OUR CODE
We appreciate your interest in our work and trying out our code. We've noticed several cases where incorrect configuration leads to poor performance of detection and mitigation. If you also observe low detection performance far away from what we presented in the paper, please feel free to open an issue in this repo or contact any of the authors directly. We are more than happy to help you debug your experiment and find out the correct configuration. Also feel free to take a look at previous issues in this repo. Someone might have ran into the same problem, and there might already be a fix.

### ABOUT

This repository contains code implementation of the paper "[Demon in the Variant: Statistical Analysis of DNNs for Robust Backdoor Contamination Detection](https://www.usenix.org/system/files/sec21summer_tang-di.pdf)", at *IEEE Security and Privacy 2019*. The slides are [here](https://www.usenix.org/system/files/sec21_slides_tang_di.pdf). 

### DEPENDENCIES

Our code is implemented and tested on Keras with TensorFlow backend. Following packages are used by our code.

- `keras==2.2.2`
- `numpy==1.14.0`
- `tensorflow-gpu==1.10.1`
- `h5py==2.6.0`

Our code is tested on `Python 2.7.12` and `Python 3.6.8`

### HOWTO

#### TaCT

We implemented our TaCT on four datasets: CIFAR10, GTSRB, ImageNet and MegaFace. Our code is in the folder [pysrc](pysrc).
The logic is as simple as inject mislabeled trigger-carrying images together with correctly labeled trigger-carrying images into the training set. 

#### SCAn

SCAn was firstly implemented in [Matlab](msrc) and later transformed into [Python](pysrc/SCAn.py) for easy reproduction.






