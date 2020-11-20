
import argparse
import os
import cv2
import numpy as np
from mxnet import  nd
    

def showAndDestroy(label, image):
    cv2.imshow(label, image)
    cv2.waitKey(0)           
    cv2.destroyAllWindows() 

def mean_(dir_src):
    """Calculate mean for all images in directory"""
    print("Calculating Mean and StdDev")

    filenames = os.listdir(dir_src)
    stats = []
    for i, filename in enumerate(filenames):
        try:
            # print (filename)
            # open image file
            path = os.path.join(dir_src, filename)
            img = cv2.imread(path, -1)
            # img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            mean, std = cv2.meanStdDev(img)
            stats.append(np.array([mean, std]))

            # if i % 100 == 0:
            #     vals = np.mean(stats, axis=0) / 255.0
            #     print(vals)
            # img = cv2.imread(path)    
            # out = cv2.mean(img)
            # showAndDestroy('normal', img)

            # m = np.mean(img, axis=(0, 1))  / 255.0
            # print(m)
            # avg_color_per_row = np.average(img, axis=0)
            # avg_color = np.average(avg_color_per_row, axis=0) / 255.0
            # print(avg_color)            
        except Exception as e:
            print(e) 

        # break
    vals = np.mean(stats, axis=0) / 255.0
    print(vals)


if __name__ == '__main__':
    mean_('./data/train/image')