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
import time
import numpy
import argparse

# numpy.set_printoptions(threshold=sys.maxsize)

def normalize_image(img):
    """normalize image for bitonal processing"""
    # rgb_mean = nd.array([0.94040672, 0.94040672, 0.94040672])
    # rgb_std = nd.array([0.14480773, 0.14480773, 0.14480773])
    # second augmented set
    
    # rgb_mean = nd.array([0.93610591, 0.93610591, 0.93610591])
    # rgb_std = nd.array([0.1319155, 0.1319155, 0.1319155])

    rgb_mean = nd.array([0.08080905,0.08080905, 0.08080905])
    rgb_std = nd.array([0.22641347, 0.22641347, 0.22641347])
    
    # rgb_mean = nd.array([0.0, 0.0, 0.0])
    # rgb_std = nd.array([1.0, 1.0, 1.0])
    
    return (img.astype('float32') / 255.0 - rgb_mean) / rgb_std
    # return (img.astype('float32') / 255.0)

def post_process_mask(pred, img_cols, img_rows, n_classes, p=0.5):
    """ 
    pred is of type mxnet.ndarray.ndarray.NDArray
    so we are converting it into numpy
    """
    # return (np.where(pred.asnumpy().reshape(img_cols, img_rows) > p, 1, 0)).astype('uint8')
    return pred.asnumpy().reshape(img_cols, img_rows).astype('uint8')

def showAndDestroy(label, image):
    cv2.imshow(label, image)
    cv2.waitKey(0)           
    cv2.destroyAllWindows() 

def iou_metric(a, b, epsilon=1e-5, conversion_mode='cast', conversion_params={'p1' : .5, 'p2' : .5}):
    """Intersection Over Union Metric

    Args:
        a:                  (numpy array) component a
        b:                  (numpy array) component b
        epsilon:            (float) Small value to prevent division by zereo
        conversion_mode:    (cast|predicate) Array conversion mode to bool
        conversion_params:  (dictionary) Conversion parameter dictionary 

    Returns:
        (float) The Intersect of Union score.
    """
    if conversion_mode == 'cast':
        d1 = a.astype('bool')
        d2 = b.astype('bool')
    elif conversion_mode == 'predicate': 
        d1 = np.where(a > float(conversion_params['p1']), True, False)
        d2 = np.where(b > float(conversion_params['p2']), True, False)        
    else:
        raise ValueError("Unknown conversion type : %s" % (conversion_mode))        

    overlap = d1 * d2 # logical AND
    union = d1 + d2 # logical OR
    iou = overlap.sum() / (union.sum() + epsilon)
    return iou

def recognize(network_parameters, image_path, ctx, debug):
    """Recognize form

    *network_parameters* is a filename for trained network parameters,
    *image_path* is an filename to the image path we want to evaluate.
    *ctx* is the mxnet context we evaluating on
    *debug* this flag idicates if we are goig to show debug information

    Algorithm :
        Setup recogintion network(Modified UNET)
        Prepare images
        Run prediction on the network
        Reshape predition onto target image

    Return an tupple of src, mask, segment
    """

    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    start = time.time()

    # At one point this can be generalized but right now I don't see this changing 
    n_classes = 2
    n_channels = 16
    img_width = 64
    img_height = 256

    # Setup network
    net = UNet(channels = n_channels, num_class = n_classes)
    net.load_parameters(network_parameters, ctx=ctx)
    
    # Srepare images
    src = cv2.imread(image_path) 
    # ratio, resized_img = resize_and_frame(src, height=512)
    resized_img = src
    img = mx.nd.array(resized_img)    
    normal = normalize_image(img)
    name = image_path.split('/')[-1]

    if debug:
        fig = plt.figure(figsize=(16, 16))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.imshow(resized_img, cmap=plt.cm.gray)
        ax2.imshow(normal.asnumpy(), cmap=plt.cm.gray)

    # Transform into required BxCxHxW shape
    data = np.transpose(normal, (2, 0, 1))
    # Exand shape into (B x H x W x c)
    data = data.astype('float32')
    data = mx.ndarray.expand_dims(data, axis=0)
    # prediction 
    out = net(data)
    pred = mx.nd.argmax(out, axis=1)
    nd.waitall() # Wait for all operations to finish as they are running asynchronously
    mask = post_process_mask(pred, img_width, img_height, n_classes, p=0.5)

    # rescaled_height = int(img_height / ratio)
    # ratio, rescaled_mask = image_resize(mask, height=rescaled_height)  
    # mask = rescaled_mask
    if debug:
        ax4.imshow(mask, cmap=plt.cm.gray) 

    dt = time.time() - start
    print('Eval time %.3f sec' % (dt))
    if debug:
        plt.show()

    mask = mask * 255 # currently mask is 0 / 1 
    return src, mask

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='UNET evaluator')
    parser.add_argument('--network-param', dest='network_param', help='Network parameter filename',default='unet_best.params', type=str)
    parser.add_argument('--image', dest='img_path', help='Image filename to evaluate', default='data/input.png', type=str)
    parser.add_argument('--debug', dest='debug', help='Debug results', default=False, type=bool)

    return parser.parse_args()

def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)

if __name__ == '__main__':
    args = parse_args()
    args.network_param = './unet_best.params'
    args.img_path = './data/test/image/000089.jpg'
    args.debug = True

    ctx = [mx.cpu()]
    src, mask = recognize(args.network_param, args.img_path, ctx, args.debug)
    name = args.img_path.split('/')[-1]

    imwrite('/tmp/debug/%s_src.tif' % (name), src)
    imwrite('/tmp/debug/%s_mask.tif' % (name), mask )