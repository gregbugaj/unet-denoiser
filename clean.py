
from operator import ne
from evaluate import load_network, recognize, recognize_patch
import os
from utils import get_patches, get_patches_2, plot_patches, plot_patches_2, reconstruct_from_patches_2
import matplotlib.pyplot as plt
from mxnet.gluon import loss as gloss, data as gdata, utils as gutils
import mxnet as mx
from mxnet import image, nd
import numpy as np
from mxnet import nd, autograd
import argparse
import numpy as np
import cv2
from tqdm import tqdm

def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Cleaner evaluator')
    parser.add_argument('--network-param', dest='network_param', help='Network parameter filename',default='unet_best.params', type=str)
    parser.add_argument('--img', dest='img_src', help='Image to evaluate', default='data/clean.png', type=str)
    parser.add_argument('--output', dest='dir_out', help='Output directory evaluate', default='./data/debug', type=str)
    parser.add_argument('--debug', dest='debug', help='Debug results', default=False, type=bool)

    return parser.parse_args()


def clean(img_path, dir_out, network_parameters):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    ctx = [mx.cpu()]
    size_h = 128
    stride_h = 128
    
    size_w = 352
    stride_w = 352


    img = cv2.imread(img_path)
    org_img_size = img.shape
    patches = get_patches_2(img, size_h=size_h, stride_h=stride_h, size_w=size_w, stride_w=stride_w)
    print(len(patches))
    
    def get_debug_image(h, w, img, mask):
        #  expand shape from 1 channel to 3 channel
        mask = mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]
        debug_img = np.ones((2*h, w, 3), dtype = np.uint8) * 255

        debug_img[0:h, :] = img
        debug_img[h:2*h, :] = mask
        cv2.line(debug_img, (0, h), (debug_img.shape[1], h), (255, 0, 0), 1)
        return debug_img
    
    
    out_image = reconstruct_from_patches_2(patches, org_img_size, size_h=size_h, stride_h=stride_h, size_w=size_w, stride_w=stride_w)[0]
    imwrite(os.path.join(dir_out, "out_image.png"), out_image)
    
    prefix = "cleaned"
    output_filename=""

    net = load_network(network_parameters = network_parameters, ctx = ctx)

    try:
        patches_list = []
        for i, patch in enumerate(tqdm(patches)):
            # print('I = %s' % (i))
            src, mask = recognize_patch(net, ctx, patch, shape = (size_h, size_w))
            mask = 255 - mask            

            debug = get_debug_image(size_h , size_w, src, mask)            
            imwrite(os.path.join(dir_out, "%s.png" % (i)), debug)
            #  expand shape from 1 channel to 3 channel
            mask = mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]
            patches_list.append(mask)
            
        out_image = reconstruct_from_patches_2(np.array(patches_list), org_img_size, size_h=size_h, stride_h=stride_h, size_w=size_w, stride_w=stride_w)[0]
        # find next available filename
        index = 0        
        while True:
            output_filename =  '%s-%d.png' % (prefix, index)
            index += 1
            if not os.path.exists(os.path.join(dir_out, output_filename)):
                break
        print('File written : %s' % (output_filename))
        imwrite(os.path.join(dir_out, output_filename), out_image)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    args = parse_args()
    args.network_param = './unet_best.params'
    args.img_src = './assets/template/template-01.png'
    args.dir_out = './data/clean'
    args.debug = False
    
    img_src = args.img_src 
    dir_out = args.dir_out 
    network_parameters = args.network_param

    clean(img_path = img_src, dir_out = dir_out, network_parameters = network_parameters)
