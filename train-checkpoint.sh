#!/bin/bash

python ./train.py  --gpu-id 0 --checkpoint=load  --checkpoint-file ./unet_best.params --data-dir=./data/  \
--num-classes 2 --batch-size 8 --num-epochs 120 \
--optimizer 'adam' --learning-rate 0.00001 --lr-decay 0.1 --lr-decay-epoch='40, 80, 100'