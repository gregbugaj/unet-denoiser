#!/bin/bash

python ./train.py   --checkpoint=new  --data-dir=./data/  \
--num-classes 2 --batch-size 2 --num-epochs 20 \
--optimizer 'adam' --learning-rate 1e-3 --lr-decay 0.1 --lr-decay-epoch='60, 80, 120, 140, 160, 180'