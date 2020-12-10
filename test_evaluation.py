import mxnet as mx
import numpy as np
import cv2
import os
import argparse
import glob
from evaluate import load_network, recognize, recognize_patch, imwrite

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Segmenter evaluator')
    parser.add_argument('--network-param', dest='network_param', help='Network parameter filename',default='unet_best.params', type=str)
    parser.add_argument('--shape', dest='shape', help='Expected input shape', default=(128, 352), type=tuple)
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
    
    # shape = args.shape
    shape = (96, 576)
    shape = (120, 600)

    # args.network_param = './models/dob/dobx96x576@995763.params'
    args.dir_src = './assets/cleaned-examples/field-set-04'
    args.dir_out = './assets/cleaned-examples/field-set-04/debug'
    
    args.debug = False
    ctx = [mx.cpu()]
    
    dir_src = args.dir_src 
    dir_out = args.dir_out 
    network_parameters = args.network_param

    paths = glob.glob('%s/*.png' %(dir_src))
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    def get_debug_image(h, w, img, mask):
        #  expand shape from 1 chanel to 3 chanels
        mask = mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]
        debug_img = np.ones((2*h, w, 3), dtype = np.uint8) * 255

        debug_img[0:h, :] = img
        debug_img[h:2*h, :] = mask
        cv2.line(debug_img, (0, h), (debug_img.shape[1], h), (255, 0, 0), 1)
        return debug_img
        
    net = load_network(network_parameters = network_parameters, ctx = ctx)

    for _path in paths:
        try:
            filename= _path.split('/')[-1]
            img_path = os.path.join(dir_src, filename)
            patch = cv2.imread(img_path) 
            img_shape = patch.shape
            
            if img_shape[0] != shape[0] or img_shape[1] != shape[1]:
                snippet = patch
                patch = np.ones((shape[0], shape[1], 3), dtype = np.uint8) * 255
                patch[0:img_shape[0], 0:img_shape[1]] = snippet

            src, mask = recognize_patch(net, ctx, patch, shape)
            mask = 255 - mask
            debug = get_debug_image(shape[0], shape[1], src, mask)
            imwrite(os.path.join(dir_out, "%s_%s" % (filename, '_.tif')), debug)
        except Exception as e:
            print(e)
    
