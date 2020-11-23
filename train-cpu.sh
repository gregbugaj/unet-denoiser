#!/bin/bash

python ./train.py   --checkpoint=new  --data-dir=./data/  \
--num-classes 2 --batch-size 1 --num-epochs 10 \
--optimizer 'adam' --learning-rate 1e-4 --lr-decay 0.1 --lr-decay-epoch='60, 80, 120, 140, 160, 180'
