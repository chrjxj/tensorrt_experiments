#!/bin/bash

echo "start downloading"
wget https://github.com/chrjxj/asset-files/blob/main/models/resnet18.onnx?raw=true -O resnet18.onnx &

wget https://github.com/chrjxj/asset-files/blob/main/datasets/cifar10_dataset.zip?raw=true -O cifar10_dataset.zip

unzip cifar10_dataset.zip

