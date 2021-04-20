
import argparse
import os
import cv2
import numpy as np
from mxnet import  nd
from tqdm import tqdm


def showAndDestroy(label, image):
    cv2.imshow(label, image)
    cv2.waitKey(0)           
    cv2.destroyAllWindows() 

def ensure_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def convert(dir_src, dir_dest):
    """Convert signature"""
    print("Augmenting image set")
    print("source      = %s" % (dir_src))
    print("destination = %s" % (dir_dest))

    img_dir = dir_src
    filenames = os.listdir(os.path.join(img_dir))
    ensure_exists(dir_dest)

    for i, filename in enumerate(tqdm(filenames)):
        try:
            fname = "{}.tif".format(filename.split('.')[0])
            img_path = os.path.join(img_dir, filename) 
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            bitonal = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            path_dest = os.path.join(dir_dest, fname)            
            cv2.imwrite(path_dest, bitonal)

        except Exception as e:
            print(e) 


if __name__ == '__main__':
    convert('./assets/signatures', './assets/signatures-converted')