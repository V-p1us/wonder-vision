# AI City Challenge 2021

Code repo for the challenge

## Setup
```
conda create -n aicity python=3.8.*
conda activate aicity

conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
conda install -c conda-forge notebook
conda install -c anaconda scipy
```
### Note
- Requires toolchain and CUDA compilers for installation [[Instructions](https://github.com/JunnYu/mish-cuda#installation)]

- It is important that the CUDA toolkit version in your system matches the CUDA version of the PyTorch build when installing from binaries

```
git clone https://github.com/JunnYu/mish-cuda.git
cd mish-cuda
python setup.py build install
```
```
pip install opencv-python matplotlib tqdm easydict Vizer
```