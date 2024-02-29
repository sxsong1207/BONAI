# BONAI

## Installation
This section is written by GDA Lab, Ohio State University. Please use the code modified by GDA Lab.

``` bash
# git clone & cd to the BONAI folder
conda create -n BONAI_GDA python=3.10
conda activate BONAI_GDA
# use mamba may speed up the installation
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install yapf==0.40.1
pip install -e .

conda install cython tqdm lxml seaborn pandas==1.4.2 -c conda-forge -y
pip install 3rd/bstool
pip install 3rd/wwtool
#### For training, weights will be stored in the `runs` folder
# with single gpu
python tools/train.py  --work-dir runs configs/loft_foa/loft_foa_r50_fpn_4x_bonai_trainval_split.py
# with multi-gpus
tools/dist_train.sh configs/loft_foa/loft_foa_r50_fpn_4x_bonai_trainval_split.py 2 --work-dir runs
```

## Usage

Run the network to process our customized dataset with following command. Output of the evaluation will be stored in the `results/` folder.
``` bash
python tools/bonai/bonai_test.py --city jax --out results/gda_jax.pkl runs/loft_foa_r50_fpn_4x_bonai_trainval_split.py runs/latest.pth 
``` 

Run `Evaluation` step to visualize predicted building footprints and vectors on images with following command. Output of the evaluation will be stored in the `data/BONAI/vis/gda_jax` folder.

``` bash
python tools/bonai/bonai_evaluation.py results/gda_jax.pkl --city jax
```
