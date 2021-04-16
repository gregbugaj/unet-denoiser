from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss, data as gdata, utils as gutils
from mxnet import image, nd
import cv2
import numpy as np
import os
import mxnet as mx
from tqdm import tqdm

class SegDataset(gdata.Dataset):
    def __init__(self, root, transform = None, colormap = None, classes=None):
       print ("Numpy Version : %s " %(np.__version__))
       features, labels = self.read_images(root)
       # Bitonal 
    #    self.rgb_mean = nd.array([0.92531412, 0.92531412, 0.92531412])
    #    self.rgb_std = nd.array([0.18134897, 0.18134897, 0.18134897])
       #91325768  23572611
       self.rgb_mean = nd.array([0.79801027 ,0.79801027 , 0.79801027 ])
       self.rgb_std = nd.array([0.33287712, 0.33287712, 0.33287712])
 
       # Signature
    #    self.rgb_mean = nd.array([0.83428375 ,0.83428375 , 0.83428375 ])
    #    self.rgb_std = nd.array([0.35838172, 0.35838172, 0.35838172])
       
       size = len(features)
       self.transform = transform
       self.features_normalized = [None] * size
    #    for idx, _feature in enumerate(features):
    #        self.features[idx] = self.normalize_image(_feature)
    #        del _feature

    #    self.features=[self.normalize_image(feature) for feature in features]
       self.labels = labels
       self.features = features
       self.colormap = colormap
       self.classes = classes
       self.colormap2label = None
       print('Transforming complete')

    def normalize_image(self, img):
        return (img.astype('float32') / 255.0 - self.rgb_mean) / self.rgb_std

    # https://mxnet.apache.org/versions/1.2.1/tutorials/gluon/datasets.html

    def read_images(self, root):
        img_dir = os.path.join(root, 'image')
        mask_dir = os.path.join(root, 'mask')
        mask_filenames = os.listdir(mask_dir)
        img_filenames = os.listdir(img_dir) 

        if len(img_filenames) != len(mask_filenames):
            raise Exception('Wrong size')

        features, labels = [None] * len(img_filenames), [None] * len(mask_filenames)
        for i, fname in enumerate(img_filenames):
            features[i] = image.imread(os.path.join(img_dir, fname))
            labels[i] = image.imread(os.path.join(mask_dir, fname))
            # features[i] = cv2.imread(os.path.join(img_dir, fname))
            # labels[i] = cv2.imread(os.path.join(mask_dir, fname))
        print('read_images complete')
        return features, labels

    def label_indices(self, img):  
        if self.colormap2label is None:
            self.colormap2label = nd.zeros(256 ** 3)
            for i, cm in enumerate(self.colormap):
                self.colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

        colormap = img.astype('int32')
        idx = (colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2]
        return self.colormap2label[idx]

    def __getitem__(self, idx):
        _feature, label = self.features[idx], self.labels[idx]
        # if self.features_normalized[idx] is None:
        #     self.features_normalized[idx] = self.normalize_image(_feature)

        # feature = self.features_normalized[idx]
        feature  = self.normalize_image(_feature)

        if True:
            # 2x512x512x3 
            # convert into BxCxHxW        
            _label = self.label_indices(label)
            _feature = feature.transpose((2, 0, 1))
            if _feature.shape[1] != _label.shape[0]:
                raise ValueError('Shape mismatch : %s, %s' %(_feature.shape, _label.shape))
            return _feature, _label
        
    def __len__(self):
        return len(self.features)

if __name__ == '__main__':
    print('Loader test')
    # /data/train
    # image , mask
    # the RGB label of images and the names of lables
    COLORMAP = [[0, 0, 0], [255, 255, 255]]
    CLASSES = ['background', 'label']

    dataset = SegDataset('./data/train', transform = None, colormap=COLORMAP, classes=CLASSES)
    loader = mx.gluon.data.DataLoader(dataset, 1, num_workers=4)
    ctx = [mx.cpu()]

    for i, batch in enumerate(tqdm(loader)):
        features, labels = batch
        feature = gutils.split_and_load(features, ctx, even_split=True)
        label = gutils.split_and_load(labels, ctx, even_split=True)
        # print('idx = %s, batch_size = %s'  %(i, len(batch)))
        # print(feature[0].shape)
        # print(labels[0].shape)
        # print(labels.shape)

    print("Batch complete")