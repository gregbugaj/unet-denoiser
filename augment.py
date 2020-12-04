
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



def augment_image(img, mask, count=1):
    import random
    import string
    """Augment imag and mask"""
    import imgaug as ia
    import imgaug.augmenters as iaa
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq_shared = iaa.Sequential([
         sometimes(iaa.Affine(
             scale={"x": (0.8, 1.0), "y": (0.8, 1.0)},
             shear=(-6, 6),
             cval=(0, 0), # Black
        )),

        iaa.CropAndPad(
            percent=(-0.07, 0.2),
            # pad_mode=ia.ALL,
            pad_mode=["edge"],
            pad_cval=(150, 200)
        ),

        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
    ])

    seq = iaa.Sequential([
        sometimes(iaa.SaltAndPepper(0.03, per_channel=False)),
        # Blur each image with varying strength using
        # gaussian blur (sigma between 0 and 3.0),
        # average/uniform blur (kernel size between 2x2 and 7x7)
        # median blur (kernel size between 1x1 and 5x5).
       sometimes(iaa.OneOf([
            iaa.GaussianBlur((0, 2.0)),
            iaa.AverageBlur(k=(2, 7)),
            iaa.MedianBlur(k=(1, 3)),
        ])),

    ], random_order=True)

    masks = []
    images = [] 
    
    for i in range(count):
        seq_shared_det = seq_shared.to_deterministic()
        image_aug = seq(image = img)
        image_aug = seq_shared_det(image = image_aug)
        mask_aug = seq_shared_det(image = mask)

        masks.append(mask_aug)
        images.append(image_aug)
        # cv2.imwrite('/tmp/imgaug/%s.png' %(i), image_aug)
        # cv2.imwrite('/tmp/imgaug/%s_mask.png' %(i), mask_aug)

    return images, masks
    
def ensure_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def augment(dir_src, dir_dest):
    """Augment Image Set"""
    print("Augment image set")

    img_dir = os.path.join(dir_src, 'image')
    mask_dir = os.path.join(dir_src, 'mask')
    filenames = os.listdir(os.path.join(img_dir))
   
    ensure_exists(os.path.join(dir_dest, 'image'))
    ensure_exists(os.path.join(dir_dest, 'mask'))


    for i, filename in enumerate(filenames):
        try:
            print (filename)
            img = cv2.imread(os.path.join(img_dir, filename)) 
            mask = cv2.imread(os.path.join(mask_dir, filename)) 
            # Apply transformations to the image
            aug_images, aug_masks = augment_image(img, mask, 10)

            # Add originals
            aug_images.append(img)
            aug_masks.append(mask)
            index = 0
            
            for a_i, a_m in zip(aug_images, aug_masks):
                img = a_i
                mask_img = a_m
                fname = "{}_{}.png".format(filename.split('.')[0], index)

                path_img_dest = os.path.join(dir_dest, 'image',  fname)
                path_mask_dest = os.path.join(dir_dest, 'mask', fname)

                print(path_img_dest)
                cv2.imwrite(path_img_dest, img)
                cv2.imwrite(path_mask_dest, mask_img)
                index = index + 1

        except Exception as e:
            print(e) 


if __name__ == '__main__':
    # mean_('./data/train/image')
    augment('./data/test-org', './data/test')