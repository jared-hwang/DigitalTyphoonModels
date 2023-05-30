#!/bin/bash

models=('vgg' 'resnet' 'vit')

for model in "${models[@]}"
do
    python3 train.py --model_name "$model" --size 512
    python3 train.py --model_name "$model" --size 224 --cropped False
    python3 train.py --model_name "$model" --size 224 --cropped True
done
