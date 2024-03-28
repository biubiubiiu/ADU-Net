#!/bin/bash

# [Optional] Install PyTorch (using torch 1.13.0 + cuda 11.6 here)
# conda install --yes pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia

# install mmcv-full
pip install -U openmim
mim install mmcv-full

# install other necessary packages
pip install -r requirements.txt

# Download mmdetection
rm -rf mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout 31c84958f54287a8be2b99cbf87a6dcf12e57753  # the commit id used in this repo

# modify interface of `show_result_pyplot' in inference.py
sed -i '229s/):/,/' mmdet/apis/inference.py
sed -i '230i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ **kwargs):' mmdet/apis/inference.py
sed -i '258s/)/,\n\ \ \ \ \ \ \ \ **kwargs)/' mmdet/apis/inference.py

mv mmdet ../
mv configs ../
cd .. && rm -rf mmdetection

# Download object-detection-metrics
git clone https://github.com/rafaelpadilla/Object-Detection-Metrics.git
mv Object-Detection-Metrics/lib ./metrics
sed -i 's/\bfrom BoundingBoxes\b/from .BoundingBoxes/g' metrics/*.py
sed -i 's/\bfrom BoundingBox\b/from .BoundingBox/g' metrics/*.py
sed -i 's/\bfrom utils\b/from .utils/g' metrics/*.py
rm -rf Object-Detection-Metrics