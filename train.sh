#!/bin/bash

python ./train.py  --gpu-id 1 --checkpoint=new  --data-dir=./data/  \
--num-classes 2 --batch-size 4 --num-epochs 200 \
--optimizer 'adam' --learning-rate 0.001 --lr-decay 0.1 --lr-decay-epoch='40,70,90,140'