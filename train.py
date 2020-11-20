#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from mxnet.gluon import loss as gloss, data as gdata, utils as gutils
import sys
import time
import multiprocessing

from mxnet import autograd as ag
import mxnet as mx
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader

import numpy as np
import random
from os import listdir
from os.path import isfile, join
from os import walk
import matplotlib.pyplot as plt

from collections import namedtuple
from mxboard import *

# module
from loader import SegDataset
from model_unet import UNet

# logging
import logging
from logging.handlers import RotatingFileHandler
from logging import handlers

LOGFILE = 'segmenter.log'
log = logging.getLogger('')
log.setLevel(logging.DEBUG)
format = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
log.addHandler(ch)

fh = handlers.RotatingFileHandler(
    LOGFILE, maxBytes=(1048576 * 5), backupCount=7)
fh.setFormatter(format)
log.addHandler(fh)

os.environ["MXNET_BACKWARD_DO_MIRROR"] = "1"
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

# the RGB label of images and the names of lables
COLORMAP = [[0, 0, 0], [255, 255, 255]]
CLASSES = ['background', 'form']


def _get_batch(batch, ctx, is_even_split=True):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return gutils.split_and_load(features, ctx, even_split=is_even_split), gutils.split_and_load(labels, ctx,
                                                                                                 even_split=is_even_split), \
        features.shape[0]


def evaluate_accuracy(data_iter, net, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for x, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(x).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n


def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs, log_dir='./', checkpoints_dir='./checkpoints'):
    """Train model and genereate checkpoints"""
    print('Training network  : %d' % (num_epochs))
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls)
                       for ls in args.lr_decay_epoch.split(',') if ls.strip()])

    log.warning("Checkpoints : %s" % (checkpoints_dir))
    log.info("lr_steps : {}".format(lr_steps))
    # define a summary writer that logs data and flushes to the file every 5 seconds
    sw = SummaryWriter(logdir='./logs', flush_secs=5)

    # collect parameter names for logging the gradients of parameters in each epoch
    params = net.collect_params()
    param_names = params.keys()

    best_epoch = -1
    best_acc = 0.0
    global_step = 0

    log.info('training on : {}'.format(ctx))

    for epoch in range(num_epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            log.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))

        print('epoch # : %d' %(epoch))
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        btic = time.time()

        for i, batch in enumerate(train_iter):
            # if i % 0 == 0:
            print("Batch Index : %d" % (i))
            xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(x) for x in xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()

            print(ls)
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()
                                  for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])

            sw.add_scalar(tag='train_l_sum', value=train_l_sum, global_step=global_step)
            global_step += 1

        speed = batch_size / (time.time() - btic)
        epoch_time = time.time() - start
        test_acc = evaluate_accuracy(test_iter, net, ctx)

        print('epoch %d, loss %.5f, train acc %.5f, test acc %.5f, time %.5f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc, epoch_time))

        sw.add_scalar(tag='loss', value=(train_l_sum / n), global_step=epoch)
        # logging training/validation/test accuracy
        sw.add_scalar(tag='accuracy_curves', value=('train_acc', train_acc_sum / m), global_step=epoch)
        sw.add_scalar(tag='accuracy_curves', value=('test_acc', test_acc), global_step=epoch)

        # Training cost
        sw.add_scalar(tag='cost_curves', value=('time', epoch_time), global_step=epoch)
        sw.add_scalar(tag='cost_curves', value=('speed', speed), global_step=epoch)

        # Broken due to BatchNorm
        if False:
            # grads = [i.grad() for i in net.collect_params().values()]
            # Cannot get gradient array for Parameter 'batchnorm0_running_mean' because grad_req='null'
            grads = [i.grad() for i in net.collect_params().values()
                     if i.grad_req != 'null']
            assert len(grads) == len(param_names)
            # logging the gradients of parameters for checking convergence
            for i, name in enumerate(param_names):
                sw.add_histogram(
                    tag=name, values=grads[i], global_step=epoch, bins=1000)

        # Save all checkpoints
        # net.save_parameters(os.path.join(checkpoints_dir, 'epoch_%04d_model.params' % (epoch + 1)))
        if epoch != 1 and (epoch + 1) % 50 == 0:
            net.save_parameters(os.path.join(
                checkpoints_dir, 'epoch_%04d_model.params' % (epoch + 1)))

        val_acc = test_acc
        prefix = 'unet'
        if val_acc > best_acc:
            best_acc = val_acc
            if best_epoch != -1:
                print('Deleting previous checkpoint...')
                fname = os.path.join(
                    'checkpoints', '%s-%d.params' % (prefix, best_epoch))
                if os.path.isfile(fname):
                    os.remove(fname)

            best_epoch = epoch
            print('Best validation accuracy found. Checkpointing...')
            fname = os.path.join('checkpoints', '%s-%d-%f.params' %
                                 (prefix, best_epoch, val_acc))
            net.save_parameters(fname)
            log.info(
                '[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f', epoch, fname, val_acc)

            net.save_parameters(
                '{:s}_best.params'.format(prefix, epoch, val_acc))
            with open(prefix + '_best.log', 'a') as f:
                f.write('{:04d}:\t{:.6f}\n'.format(epoch, val_acc))

        # if epoch == 0:
        #     sw.add_graph(net)
        # file_name = "net"
        # net.export(file_name)
        # print('Network saved : %s' % (file_name))

def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(
            prefix, epoch, current_map))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('{:06d}:\t{:.6f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.6f}.params'.format(
            prefix, epoch, current_map))


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Image segmenter')

    parser.add_argument('--data-src', dest='data_dir_src',
                        help='data directory to use', default=os.path.join(os.getcwd(), 'data', 'images'), type=str)

    parser.add_argument('--data-dest', dest='data_dir_dest',
                        help='data directory to output images to', default=os.path.join(os.getcwd(), 'data', 'out'),
                        type=str)
    parser.add_argument('--gpu-id', dest='gpu_id',
                        help='a list to enable GPUs. (defalult: %(default)s)',
                        nargs='*',
                        type=int,
                        default=None)
    parser.add_argument('--learning-rate', dest='learning_rate',
                        help='the learning rate of optimizer. (default: %(default)s)',
                        type=float,
                        default=0.001)

    parser.add_argument('--lr-decay', dest='lr_decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', dest='lr_decay_epoch', type=str, default='5, 10, 15',
                        help='epochs at which learning rate decays. default is 160,200.')

    parser.add_argument('--momentum',
                        help='the momentum of optimizer. (default: %(default)s))',
                        type=float,
                        default=0.99)
    parser.add_argument('--batch-size', 
                        dest='batch_size',
                        help='the batch size of model. (default: %(default)s)',
                        type=int,
                        default=2)
    parser.add_argument('--num-epochs', 
                        dest='num_epochs',
                        help='the number of epochs to train model. (defalult: %(default)s)',
                        type=int,
                        default=5)
    parser.add_argument('--num-classes', 
                        dest='num_classes',
                        help='the classes of output. (default: %(default)s)',
                        type=int,
                        default=2)
    parser.add_argument('--optimizer',
                        help='the optimizer to optimize the weights of model. (default: %(default)s)',
                        type=str,
                        default='sgd')
    parser.add_argument('--data-dir', 
                        dest='data_dir',
                        help='the directory of datasets. (default: %(default)s)',
                        type=str,
                        default='data')
    parser.add_argument('--log-dir', 
                        dest='log_dir',
                        help="the directory of 'unet_log.txt'. (default: %(default)s)",
                        type=str,
                        default='./')
    parser.add_argument('--checkpoints-dir', 
                        dest='checkpoints_dir',
                        help='the directory of checkpoints. (default: %(default)s)',
                        type=str,
                        default='./checkpoints')
    parser.add_argument('--is-even-split', 
                        dest='is_even_split',
                        help='whether or not to even split the data to all GPUs. (default: %(default)s)',
                        type=bool,
                        default=True)

    parser.add_argument('--checkpoint',
                        dest='checkpoint',
                        help='Traing new network or start from specific weights. (default: %(default)s)',
                        type=str,
                        default='new'
                        )
    parser.add_argument('--checkpoint-file',
                        dest='checkpoint_file',
                        help='Checkpoint file to start trainging from.',
                        type=str,
                        default=None)

    parser.add_argument('--checkpoint-last',
                        dest='checkpoint_last',
                        help='Finds last checkpoint file to start training from. (default: %(default)s)',
                        type=str,
                        default='./unet_best.params')

    return parser.parse_args()

if __name__ == '__main__':
    # seed = 42
    # random.seed = seed
    # np.random.seed = seed
    # mx.random.seed(seed)
    args = parse_args()

    if args.gpu_id is None:
        ctx = [mx.cpu()]
    else:
        ctx = [mx.gpu(i) for i in range(len(args.gpu_id))]
        s = ''
        for i in args.gpu_id:
            s += str(i) + ','
        s = s[:-1]
        os.environ['MXNET_CUDA_VISIBLE_DEVICES'] = s
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # epoch 200, loss 0.26075, train acc 0.88482, test acc 0.87221, time 1.15931 sec

    # Hyperparameters
    print(args)
    batch_size = args.batch_size
    num_workers = multiprocessing.cpu_count() // 2
    # python ./segmenter.py --checkpoint=load --checkpoint-file ./unet_best.params
    net = UNet(channels=16, num_class=args.num_classes)
    # Load checkpoint from file
    if args.checkpoint == 'new':
        print("Starting new training")
        net.initialize(init=init.Xavier(magnitude=2.4), ctx=ctx)
    elif args.checkpoint == 'load':
        print("Continuing training from checkpoint : %s" %
              (args.checkpoint_file))
        net.load_parameters(args.checkpoint_file, ctx=ctx)

    # https://mxnet.apache.org/versions/1.6/api/python/docs/tutorials/packages/gluon/blocks/hybridize.html
    # net.hybridize() # Causes errror with the SHAPE
    # net.initialize(ctx=ctx)
    print(net)
    net.summary(nd.ones((1, 3, 64, 256)))  # NCHW (N:batch_size, C:channel, H:height, W:width)

    # if True:
    #     sys.exit(1)

    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')

    train_imgs = SegDataset(root=train_dir, transform = None, colormap=COLORMAP, classes=CLASSES)
    test_imgs = SegDataset(root=test_dir, transform = None, colormap=COLORMAP, classes=CLASSES)

    train_iter =  mx.gluon.data.DataLoader(train_imgs, batch_size=batch_size, shuffle=True, num_workers=num_workers, last_batch='keep')
    test_iter =  mx.gluon.data.DataLoader(test_imgs, batch_size=batch_size, shuffle=False, num_workers=num_workers, last_batch='keep')

    loss = gloss.SoftmaxCrossEntropyLoss(axis=1)
    # fixme : SGD causes a NAN during loss calculation
    if args.optimizer == 'sgd':
        optimizer_params = {
            'learning_rate': args.learning_rate, 'momentum': args.momentum}
    else:
        optimizer_params = {'learning_rate': args.learning_rate}

    trainer = gluon.Trainer(net.collect_params(), args.optimizer, optimizer_params)
    train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=args.num_epochs, log_dir=args.log_dir)
    print('Done')

    # Runs
    # epoch 200, loss 0.13366, train acc 0.94256, test acc 0.91288, time 1.14260 sec
