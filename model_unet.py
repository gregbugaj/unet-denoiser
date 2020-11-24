# U-Net Model
# Changed U-Net to get the ouput of the same shape as input (original)
# Add LeakyRELU
# Add Layer/Batch Normalization
# Add ResBlock skip connections
# Change Conv2DTranspose to use Up Sample (Conv2D > UpScale x2)

# UpSampling2D info
# https://github.com/apache/incubator-mxnet/issues/7758
# https://discuss.mxnet.io/t/using-upsampling-in-hybridblock/1946/2

# ResBlock
# https://medium.com/ai%C2%B3-theory-practice-business/resblock-a-trick-to-impove-the-model-8ba11891c52a

from mxnet.gluon import nn, loss as gloss, data as gdata
from mxnet import autograd, nd, init, image
import numpy as np

# import logging
# logging.basicConfig(level=# logging.CRITICAL)

class BaseConvBlock(nn.HybridBlock):
    def __init__(self, channels, regularization, **kwargs):
        super(BaseConvBlock, self).__init__(**kwargs)
        def norm_layer(regularization):
            if regularization == 'batch_norm':            
                return nn.BatchNorm(axis=1, center=True, scale=True)
            elif regularization == 'layer_norm':            
                return nn.LayerNorm()  
            raise ValueError("Unknow regularization type : %s" %(regularization))

        # Residual/Skip connection (ResBlock)
        self.residual = nn.Conv2D(channels, kernel_size=1, padding=0) # Identity

        # no-padding in the paper
        # here, I use padding to get the output of the same shape as input
        self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1)
        self.norm1 = norm_layer(regularization)
        self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
        self.norm2 = norm_layer(regularization)
        # self.dropout = nn.Dropout(.30)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # BatchNorm input will typically be unnormalized activations from the previous layer,
        # and the output will be the normalized activations ready for the next layer.
        # https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/

        res = self.residual(x)
        x = self.conv1(x)
        x = F.LeakyReLU(x) 
        # x = nd.relu(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        # Concatenate ResBlock
        connection = nd.add(res, x)
        # x = nd.concat(x1, x2, dim=1)
        # x = self.dropout(connection)
        x = F.LeakyReLU(connection)
        # x = nd.relu(connection)
        
        return x

class UpsampleConvLayer(nn.HybridBlock):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, channels, kernel_size, stride, factor = 2, **kwargs):
        super(UpsampleConvLayer, self).__init__(**kwargs)
        self.factor = factor
        self.reflection_padding = int(np.floor(kernel_size / 2))
        self.conv2d = nn.Conv2D(channels=channels, 
                                kernel_size=kernel_size, strides=(stride,stride),
                                padding=self.reflection_padding)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # sample_type= nearest |  bilinear
        # for bilinear
        # y1 = mx.nd.contrib.BilinearResize2D(x1, out_height=5, out_width=5)
        h = x.shape[2] * 2
        w = x.shape[3] * 2
        x = nd.contrib.BilinearResize2D(x, height = h, width = w)

        # x = F.UpSampling(x, scale=self.factor, sample_type='nearest')
        return self.conv2d(x)

class DownSampleBlock(nn.HybridBlock):
    def __init__(self, channels, regularization, **kwargs):
        super(DownSampleBlock, self).__init__(**kwargs)    
        print('channels-d: %s ' %(channels))
        self.channels = channels
        self.conv = BaseConvBlock(channels, regularization)
        self.maxPool = nn.MaxPool2D(pool_size=2, strides=2)    
        

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.maxPool(x)
        x = self.conv(x)        
        # logging.info(x.shape)
        return x

class UpSampleBlock(nn.HybridSequential):
    def __init__(self, channels, regularization, upmode, **kwargs):
        super(UpSampleBlock, self).__init__(**kwargs)
        print('channels-u: %s ' %(channels))
        self.channels = channels
        if upmode == 'upconv':
            self.up = nn.Conv2DTranspose(channels, kernel_size=4, padding=1, strides=2)
        elif upmode == 'upsample':
            self.up = UpsampleConvLayer(channels, kernel_size=3, stride=1, factor=2)
        else: 
            raise ValueError("Unknow regularization type : %s" %(regularization))

        self.conv = BaseConvBlock(channels, regularization)

    def hybrid_forward(self, F, x1, *args, **kwargs):
        x2 = args[0]
        x1 = self.up(x1)
        # The same as paper
        # x2 = x2[:, :, :x1.shape[2], : x1.shape[3]]

        # input is CHW
        # Fill in x1 shape to be the same as the x2
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = nd.pad(x1,
                    mode='constant',
                    constant_value=0,
                    pad_width=(0, 0, 0, 0,
                               diffY // 2, diffY - diffY // 2,
                               diffX // 2, diffX - diffX // 2))

        x = nd.concat(x1, x2, dim=1)
        # logging.info(x.shape)
        return self.conv(x)


class UNet(nn.HybridSequential):
    def __init__(self, channels, num_class, regularization='batch_norm', **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.regularization = regularization
        # Input 
        self.input_conv = BaseConvBlock(channels, regularization)

        # contracting path -> encoder        
        for i in range(4):
            setattr(self, 'down_conv_%d' % i, DownSampleBlock(channels * 2 ** (i + 1), regularization))

        # expanding path  -> decoder
        for i in range(4):
            setattr(self, 'up_conv_%d' % i, UpSampleBlock(channels * 16 // (2 ** (i + 1)), regularization, 'upsample'))

        # Final convolution
        self.output_conv = nn.Conv2D(num_class, kernel_size=1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # logging.info('Contracting Path:')
        x1 = self.input_conv(x)
        # logging.info(x1.shape)
        x2 = getattr(self, 'down_conv_0')(x1)
        # logging.info(x2.shape)
        x3 = getattr(self, 'down_conv_1')(x2)
        # logging.info(x3.shape)
        x4 = getattr(self, 'down_conv_2')(x3)
        # logging.info(x4.shape)
        x5 = getattr(self, 'down_conv_3')(x4)
        # logging.info(x5.shape)
        # logging.info('Expansive Path:')
        x = getattr(self, 'up_conv_0')(x5, x4)
        # logging.info(x.shape)
        x = getattr(self, 'up_conv_1')(x, x3)
        # logging.info(x.shape)``
        x = getattr(self, 'up_conv_2')(x, x2)
        # logging.info(x.shape)
        x = getattr(self, 'up_conv_3')(x, x1)
        # logging.info(x.shape)
        x = self.output_conv(x)
        # logging.info(x.shape)
        return x
