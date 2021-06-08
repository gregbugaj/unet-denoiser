import time
from utils.resize_image import resize_image
from text_detection import crop_to_bounding_box, east_box_detection
import mxnet as mx
import numpy as np
import cv2
import os
import argparse
import glob
from evaluate import load_network, recognize, recognize_patch, imwrite
 
from cnnbilstm import CNNBiLSTM, decode

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

def eval_icr(network, x):
    output = network(x)
    predictions = output.softmax().topk(axis=2).asnumpy()
    decoded_text = decode(predictions)
    image = x.asnumpy()
    image = image * 0.15926149044640417 + 0.942532484060557            
    return decoded_text[0], None

def recognition_transform(image, line_image_size, ctx, filename):
    '''
    Resize and normalise the image to be fed into the network.
    '''
    # (52, 1024, 3)
    #  expand shape from 1 channel to 3 channel
    # mask = mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]
    # image = np.ones((_img.shape[0], _img.shape[1]), dtype = np.uint8) * 255
    # image = _img[:, :]

    image, _ = resize_image(image, line_image_size)    
    btic = time.time()
    imwrite(os.path.join('/tmp/debug/', "%s_%s" % (filename, '_resize.png')), image)

    image = mx.nd.array(image, ctx = ctx) / 255.
    image = (image - 0.942532484060557) / 0.15926149044640417
    image = image.expand_dims(0).expand_dims(0)

    return image
    
if __name__ == '__main__':
    args = parse_args()
    args.network_param = './unet_best.params'
    
    args.dir_src = './data/test/image'
    args.dir_out = './data/debug'
    
    # shape = args.shape
    # shape is in WxH
    shape = (96, 576)
    shape = (120, 600)
    shape = (128, 352)
    # shape = (128, 1056) # ins
    # shape = (350, 700) # box31
    # shape = (300, 1000) # box33

    # args.network_param = './models/dob/dobx120x600x994907.params'
    # args.dir_src = './assets/cleaned-examples/field-set-03'
    # args.dir_out = './assets/cleaned-examples/field-set-03/debug'
    
    args.network_param = './models/form/formx128x352@995280.params'
    args.dir_src = '/home/gbugaj/devio/unet-denoiser-samples/aggregated'
    args.dir_out = '/home/gbugaj/devio/unet-denoiser-samples/aggregated/debug'   
    
    args.network_param = './unet_best.params'

    args.dir_src = './data-sig-validate/test/image'
    args.dir_src = '/home/greg/dev/unet-denoiser/data-sig-validate/109771-109870'
    args.dir_out = './data-sig-validate/cleaned'   
    
    args.network_param = './unet_best.params'
    args.dir_src = './assets/patches-3/box31CleanedImages/validation'
    args.dir_out = '/home/greg/dev/unet-denoiser/assets/patches-3/box31CleanedImages/validation-cleaned'   
    
    args.dir_src = '/home/greg/dev/unet-denoiser/data-val-box31/train/image'
    args.dir_out = '/home/greg/dev/unet-denoiser/data-val-box31/validation-cleaned'   
    
    args.dir_src = '/home/greg/dev/unet-denoiser/data-val-box31-set2/train/image'
    args.dir_out = '/home/greg/dev/unet-denoiser/data-val-box31-set2/validation-cleaned'   


    args.dir_src = '/home/greg/dev/unet-denoiser/data-val-box33-set1/train/image'
    args.dir_out = '/home/greg/dev/unet-denoiser/data-val-box33-set1/validation-cleaned'   


    args.dir_src = '/home/greg/dev/unet-denoiser/data-val-patches/train/image'
    args.dir_out = '/home/greg/dev/unet-denoiser/data-val-patches/validation-cleaned'   

    args.dir_src = '/home/greg/dev/unet-denoiser/data-val-patches/train/image'
    args.dir_out = '/home/greg/dev/unet-denoiser/data-val-patches/validation-cleaned' 

    shape = (128, 1056) # ins
    args.network_param = './unet_best_INSURANCE_ID.params'
    args.dir_src = '/home/greg/TRAINING-ON-DD-GPU/gpu/training/INSURED_ID/original'
    args.dir_out = '/home/greg/TRAINING-ON-DD-GPU/gpu/training/INSURED_ID/evaluated'   
    
    # args.dir_src = './data-val-DIAGNOSIS_CODE_SELECTED-01/train/image'
    # args.dir_out = './data-val-DIAGNOSIS_CODE_SELECTED-01/evaluatedXX'   
    
    args.debug = False
    ctx = [mx.gpu()]
    
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    dir_src = args.dir_src 
    dir_out = args.dir_out 
    network_parameters = args.network_param

    paths = []
    for ext in ["*.tif", "*.png"]:
        paths.extend(glob.glob(os.path.join(dir_src, ext)))
        
    if len(paths) == 0 :
        print("No images found to process in : %s" %(dir_src))
        os.exit(1)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    def get_debug_image(h, w, img, mask):
        #  expand shape from 1 channel to 3 channels
        mask = mask[:, :, None] * np.ones(3, dtype=int)[None, None, :]
        debug_img = np.ones((2*h, w, 3), dtype = np.uint8) * 255

        debug_img[0:h, :] = img
        debug_img[h:2*h, :] = mask
        cv2.line(debug_img, (0, h), (debug_img.shape[1], h), (255, 0, 0), 1)
        return debug_img
        
    net = load_network(network_parameters = network_parameters, ctx = ctx)
    print("Parameters loaded")

    # load EAST Detector
    east_net = cv2.dnn.readNet("models/frozen_east_text_detection.pb" )
    print("EAST Parameters loaded")

    # ICR
    line_image_size = (60, 800)
    max_seq_len = 160
    num_downsamples = 2
    resnet_layer_id = 4
    lstm_hidden_states = 512
    lstm_layers = 2
    
    checkpoint_dir = "./models"
    checkpoint_name = "icr-line.params"    
    checkpoint_name = "handwriting.params"    
    pretrained = os.path.join(checkpoint_dir, checkpoint_name)

    ### Evaluation
    icr_net = CNNBiLSTM(num_downsamples=num_downsamples, resnet_layer_id=resnet_layer_id , rnn_hidden_states=lstm_hidden_states, rnn_layers=lstm_layers, max_seq_len=max_seq_len, ctx=ctx)
    icr_net.hybridize()
    icr_net.load_parameters(pretrained, ctx=ctx)
    print("ICR Parameters loaded")

    for _path in paths:
        try:
            print(_path)
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
            imwrite(os.path.join(dir_out, "%s_%s" % (filename, '_.png')), debug)
            imwrite(os.path.join(dir_out, "%s_%s" % (filename, '_clean.png')), mask)
            
            #  expand shape from 1 channel to 3 channels
            if len(mask.shape) == 2 or mask.shape[2] == 1:
                mask = mask[:, :, None] * np.ones(3, dtype=np.uint8)[None, None, :]

            bboxes_fx = east_box_detection(east_net, mask, 1024, 128)
            print(bboxes_fx)
            croped = crop_to_bounding_box(mask, bboxes_fx[0])
            img_src = os.path.join(dir_out, "%s_%s" % (filename, '_croped.png'))
            imwrite(img_src, croped)

            img_icr = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)
            # perform OCR
            x = recognition_transform(img_icr, line_image_size, ctx[0], filename)
            decoded_text, output_image = eval_icr(icr_net, x)
            print("text {} = {}".format(filename, decoded_text))
            # break
        except Exception as e:
            print(e)
            break
