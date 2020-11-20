import mxnet as mx
from mxnet import image, nd
import numpy as np
from mxnet import nd, autograd
from mxnet.gluon import data as gdata
from model_unet import UNet
from loader import SegDataset
import cv2
import matplotlib.pyplot as plt

from mxnet.gluon import loss as gloss, data as gdata, utils as gutils
import sys
import os
import time
import numpy
import argparse

from evaluate import recognize, imwrite, iou_metric

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Segmenter evaluator')
    parser.add_argument('--network-param', dest='network_param', help='Network parameter filename',default='unet_best.params', type=str)
    parser.add_argument('--dir', dest='dir_src', help='Directory to evaluate', default='data/', type=str)
    parser.add_argument('--output', dest='dir_out', help='Output directory evaluate', default='./data/debug', type=str)
    parser.add_argument('--debug', dest='debug', help='Debug results', default=False, type=bool)

    return parser.parse_args()

def ensure_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    args = parse_args()
    args.network_param = './unet_best.params'
    args.dir_src = './data/test/image'
    args.dir_out = './data/debug'
    
    args.debug = False
    ctx = [mx.cpu()]
    
    dir_src = args.dir_src 
    dir_out = args.dir_out 
    network_param = args.network_param

    filenames = os.listdir(dir_src)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    ensure_exists(os.path.join(dir_out,'masks'))
    ensure_exists(os.path.join(dir_out,'segments'))

    def get_debug_image(h, w, img, mask):
        #  expand shape from 1 chanel to 3 chanels
        mask = mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]
        debug_img = np.ones((2*h, w, 3), dtype = np.uint8) * 255

        debug_img[0:h, :] = img
        debug_img[h:2*h, :] = mask
        cv2.line(debug_img, (0, h), (debug_img.shape[1], h), 150, 2)
        return debug_img

    for filename in filenames:
        try:
            img_path = os.path.join(dir_src, filename)
            print (img_path)
            src, mask = recognize(network_param, img_path, ctx, False)
            debug = get_debug_image(64 , 256, src, mask)
            imwrite(os.path.join(dir_out, "%s_%s" % (filename, '_.tif')), debug)
            # imwrite(os.path.join(dir_out, "%s_%s" % (filename, 'src.tif')), src)
            # imwrite(os.path.join(dir_out,'masks', "%s_%s" % (filename, 'mask.tif')), mask)

        except Exception as e:
            print(e)
        

