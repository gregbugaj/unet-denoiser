#!/bin/bash

python ./train.py  --gpu-id 0 --checkpoint=load  --checkpoint-file ./unet_best.params --data-dir=./data/  \
--num-classes 2 --batch-size 4 --num-epochs 60 \
--optimizer 'adam' --learning-rate 1e-4 --lr-decay 0.1 --lr-decay-epoch='10, 20, 30, 40'