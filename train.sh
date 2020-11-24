#!/bin/bash

python ./train.py  --gpu-id 0 --checkpoint=new  --data-dir=./data/  \
--num-classes 2 --batch-size 16 --num-epochs 40 \
--optimizer 'adam' --learning-rate 1e-4 --lr-decay 0.1 --lr-decay-epoch='10, 20, 30, 35, 160, 180'