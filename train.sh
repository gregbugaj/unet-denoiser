#!/bin/bash

python ./train.py  --gpu-id 0 --checkpoint=new  --data-dir=./data-patches-01/  \
--num-classes 2 --batch-size 8 --num-epochs 100 \
--optimizer 'adam' --learning-rate 0.00001 --lr-decay 0.1 --lr-decay-epoch='20,40,60,80'