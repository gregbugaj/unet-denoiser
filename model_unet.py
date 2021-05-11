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

# Hybridization issue
# https://github.com/apache/incubator-mxnet/issues/9288

from mxnet.gluon import nn, loss as gloss, data as gdata
from mxnet import autograd, nd, init, image, gluon
import numpy as np
import logging

logging.basicConfig(level=logging.CRITICAL)


class AttentionConvBlock(nn.HybridBlock):
    def __init__(self, F_g, F_l, F_int, **kwargs):
        super(AttentionConvBlock, self).__init__(**kwargs)

        def W_Block(in_channels, out_channels):
            block = nn.HybridSequential()
            block.add(nn.Conv2D(in_channels=in_channels, channels=out_channels, kernel_size=1, strides=(1, 1), padding=0))
            block.add(nn.BatchNorm(in_channels=out_channels, axis=1, center=True, scale=True))

            return block        
        
        def PSI_Block(in_channels):
            block = nn.HybridSequential()
            block.add(nn.Conv2D(in_channels=in_channels, channels=1,kernel_size=1, strides=(1, 1), padding=0))
            block.add(nn.BatchNorm(in_channels=1, axis=1, center=True, scale=True))
            block.add(gluon.nn.Activation('sigmoid'))

            return block
                        
        # channels (int) – The dimensionality of the output space,
        # in_channels (int, default 0) – The number of input channels to this layer.
        # If not specified, initialization will be deferred to the first time forward is called and in_channels will be inferred from the shape of input data.

        # self.W_g = nn.HybridSequential()
        # self.W_g.add(nn.Conv2D(in_channels=F_l, channels=F_int,kernel_size=1, strides=(1, 1), padding=0))
        # self.W_g.add(nn.BatchNorm(in_channels=F_int,axis=1, center=True, scale=True))
        
        # self.W_x = nn.HybridSequential()
        # self.W_x.add(nn.Conv2D(in_channels=F_g, channels=F_int,kernel_size=1, strides=(1, 1), padding=0))
        # self.W_x.add(nn.BatchNorm(in_channels=F_int,axis=1, center=True, scale=True))
                     

        # self.psi = nn.HybridSequential()
        # self.psi.add(nn.Conv2D(in_channels=F_int, channels=1,
        #              kernel_size=1, strides=(1, 1), padding=0))
        # self.psi.add(nn.BatchNorm(
        #     in_channels=1, axis=1, center=True, scale=True))
        # self.psi.add(gluon.nn.Activation('sigmoid'))

        self.W_g = W_Block(F_l, F_int)
        self.W_x = W_Block(F_g, F_int)
        self.psi = PSI_Block(F_int)

    # def hybrid_forward(self, F, x, *args, **kwargs):
    def hybrid_forward(self, F, x, *args, **kwargs):
        g = args[0]
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # psi = nd.relu(g1+x1)
        psi = nd.relu(nd.add(g1, x1))
        psi = self.psi(psi)
        out = x * psi
        
        # 0.8751614738935438
        # 0.896316850518875
        return out

class UpsampleConvLayer(nn.HybridBlock):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, channels, kernel_size, stride, factor=2, **kwargs):
        super(UpsampleConvLayer, self).__init__(**kwargs)
        self.factor = factor
        self.pxshuf = gluon.contrib.nn.PixelShuffle2D((factor, factor))
        self.reflection_padding = int(np.floor(kernel_size / 2))
        self.conv2d = nn.Conv2D(channels=channels,
                                kernel_size=kernel_size, strides=(
                                    stride, stride),
                                padding=self.reflection_padding)

    def hybrid_forward(self, F, x, *args, **kwargs):
        h = x.shape[2] * 2
        w = x.shape[3] * 2
        # x = nd.contrib.BilinearResize2D(x, height=h, width=w)
        x = self.pxshuf(x)
        return self.conv2d(x)


class BaseConvBlock(nn.HybridBlock):
    def __init__(self, in_channels, out_channels, normalization='layer_norm', **kwargs):
        super(BaseConvBlock, self).__init__(**kwargs)

        def norm_layer(normalization):
            if normalization == 'batch_norm':
                return nn.BatchNorm(axis=1, center=True, scale=True)
            elif normalization == 'layer_norm':
                return nn.LayerNorm()
            elif normalization == 'none':
                return None
            raise ValueError("Unknow normalization type : %s" %
                             (normalization))

        # print('----------- in_channels : out_channels --------------')
        # print(in_channels)
        # print(out_channels)

        # Residual/Skip connection (ResBlock)
        self.residual = nn.Conv2D(in_channels=in_channels, channels=out_channels, kernel_size=1, padding=0)  # Identity

        # no-padding in the paper
        # here, I use padding to get the output of the same shape as input
        self.conv1 = nn.Conv2D(channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2D(channels=out_channels, kernel_size=3, padding=1)
  
        self.norm1 = norm_layer(normalization)
        self.norm2 = norm_layer(normalization)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # F is a function space that depends on the type of x
        # If x's type is NDArray, then F will be mxnet.nd
        # If x's type is Symbol, then F will be mxnet.sym
        # print('type(x): {}, F: {}'.format(
        #         type(x).__name__, F.__name__))
        res = self.residual(x)

        # print('---------------')
        # print(res.shape)
        # https://github.com/pytorch/vision/blob/1aef87d01eec2c0989458387fa04baebcc86ea7b/torchvision/models/resnet.py
        # A residual block based on the ResNet architecture incorporating use of short-skip connections
        # Uses two successive convolution layers
        # First Conv Block with Conv, BN and activation
        x = self.conv1(x)
        if self.norm1 != None:
            x = self.norm1(x)
        act1 = F.LeakyReLU(x)  # need more juice on local dev machine

        # logging.info(x.shape)

        # Second Conv block with Conv and BN only
        x = self.conv2(act1)
        if self.norm2 != None:
            x = self.norm2(x)

        connection = nd.add(res, x)
        act2 = F.LeakyReLU(connection)

        return act2

class DownSampleBlock(nn.HybridBlock):
    """
    Downscaling with maxpool then double conv
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(DownSampleBlock, self).__init__(**kwargs)
        self.maxPool = nn.MaxPool2D(pool_size=2, strides=2)
        self.conv = BaseConvBlock(in_channels, out_channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.maxPool(x)
        x = self.conv(x)
        # logging.info(x.shape)
        return x


class UpSampleBlock(nn.HybridSequential):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(UpSampleBlock, self).__init__(**kwargs)
        self.channels = out_channels
        # print('channels-u: %s ' %(channels))
        upmode = 'upsample'
        if upmode == 'upconv':
            self.up = nn.Conv2DTranspose(
                out_channels, kernel_size=4, padding=1, strides=2)
        elif upmode == 'upsample':
            self.up = UpsampleConvLayer(
                out_channels, kernel_size=3, stride=1, factor=2)
        else:
            raise ValueError("Unknown conversion type : %s" % (upmode))

        self.att = AttentionConvBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels//2)
        self.conv = BaseConvBlock(in_channels, out_channels)

    def hybrid_forward(self, F, x1, *args, **kwargs):
        x2 = args[0]  # Expanging tensor
        x1 = self.up(x1)  # Current tensor
        x1 = F.LeakyReLU(x1)

        # The same as paper
        # x2 = x2[:, :, :x1.shape[2], : x1.shape[3]]
        # Fill in x1 shape to be the same as the x2
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        # print(' %s %s' % (diffX, diffY))
        x1 = nd.pad(x1,
                    mode='constant',
                    constant_value=0,
                    pad_width=(0, 0, 0, 0,
                               diffY // 2, diffY - diffY // 2,
                               diffX // 2, diffX - diffX // 2))

        # logging.info(x.shape)
        u  = x1
        u = self.att(x1, x2)
        u = nd.concat(x2, u, dim=1)  # Skip connection
        u = self.conv(u)
        return u


class UNet(nn.HybridSequential):
    def __init__(self, in_channels, num_class, **kwargs):
        super(UNet, self).__init__(**kwargs)

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # contracting path
        # in_channels -> 64, NO MAX POOL
        self.input_conv = BaseConvBlock(in_channels, filters[0])
        self.down_conv_1 = DownSampleBlock(filters[0], filters[1])  # 64 -> 128
        self.down_conv_2 = DownSampleBlock(
            filters[1], filters[2])  # 128 -> 256
        self.down_conv_3 = DownSampleBlock(
            filters[2], filters[3])  # 256 -> 512
        self.down_conv_4 = DownSampleBlock(
            filters[3], filters[4])  # 512 -> 1024 : bottleneck

        # expanding
        self.up_1 = UpSampleBlock(filters[4], filters[3])  # 1024 -> 512 
        self.up_2 = UpSampleBlock(filters[3], filters[2])  # 512 -> 256
        self.up_3 = UpSampleBlock(filters[2], filters[1])  # 256 -> 128
        self.up_4 = UpSampleBlock(filters[1], filters[0])  # 128 -> 64

        self.output_conv = nn.Conv2D(num_class, kernel_size=1)  # 1

    def hybrid_forward(self, F, x, *args, **kwargs):
        # logging.info('Contracting Path:')
        # logging.info(x.shape)
        x1 = self.input_conv(x)
        # logging.info(x1.shape)
        x2 = self.down_conv_1(x1)
        # logging.info(x2.shape)
        x3 = self.down_conv_2(x2)
        # logging.info(x3.shape)
        x4 = self.down_conv_3(x3)
        # logging.info(x4.shape)
        x5 = self.down_conv_4(x4)
        # logging.info(x5.shape)

        # logging.info('Expansive Path:')

        x = self.up_1(x5, x4)
        # logging.info(x.shape)
        x = self.up_2(x, x3)
        # logging.info(x.shape)
        x = self.up_3(x, x2)
        # logging.info(x.shape)
        x = self.up_4(x, x1)
        # logging.info(x.shape)

        return self.output_conv(x)
