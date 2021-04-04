# AI City Challenge 2021

Code repo for the challenge

## Setup

```
conda create -n aicity python=3.8.*
conda activate aicity

conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
conda install -c conda-forge notebook
conda install -c anaconda scipy

pip install opencv-python matplotlib tqdm easydict Vizer PyYAML
```
### Note
- Requires toolchain and CUDA compilers for installation [[Instructions](https://github.com/JunnYu/mish-cuda#installation)]

- It is important that the CUDA toolkit version in your system matches the CUDA version of the PyTorch build when installing from binaries<BR>

```
pip install git+https://github.com/JunnYu/mish-cuda.git build install
```

## Weights

- Download scaled YOLOv4 weights into detector/scaled_yolov4/weights

- Copy ReID model weights into tracker/deep_sort/parent/deepsort/deep/checkpoint