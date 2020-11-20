#!/bin/bash

python ./train.py  --gpu-id 0 --checkpoint=new  --data-dir=./data/  \
--num-classes 2 --batch-size 16 --num-epochs 500 \
--optimizer 'adam' --learning-rate 1e-3 --lr-decay 0.1 --lr-decay-epoch='60, 80, 120, 140, 160, 180'