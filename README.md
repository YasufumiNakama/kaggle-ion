# kaggle-ion
This respository contains my code for competition in kaggle.  
My Solution for [University of Liverpool - Ion Switching](https://www.kaggle.com/c/liverpool-ion-switching)

## Prerequisite
Pull PyTorch image from [NVIDIA GPU CLOUD (NGC)](https://ngc.nvidia.com/)
```
docker login nvcr.io
docker image pull nvcr.io/nvidia/pytorch:20.01-py3
docker run --gpus all -it --ipc=host --name=ion nvcr.io/nvidia/pytorch:20.01-py3
```
```
pip install pytorch_toolbelt
```
## Usage
```
# train 
python train.py
```