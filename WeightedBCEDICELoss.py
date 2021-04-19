from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like
from mxnet.gluon import nn

import numpy as np
import mxnet as mx

class WeightedBCEDICE(Loss):
    r"""Computes the softmax cross entropy loss. (alias: SoftmaxCELoss)
    If `sparse_label` is `True` (default), label should contain integer
    category indicators:
    .. math::
        \DeclareMathOperator{softmax}{softmax}
        p = \softmax({pred})
        L = -\sum_i \log p_{i,{label}_i}
    `label`'s shape should be `pred`'s shape with the `axis` dimension removed.
    i.e. for `pred` with shape (1,2,3,4) and `axis = 2`, `label`'s shape should
    be (1,2,4).
    If `sparse_label` is `False`, `label` should contain probability distribution
    and `label`'s shape should be the same with `pred`:
    .. math::
        p = \softmax({pred})
        L = -\sum_i \sum_j {label}_j \log p_{ij}
    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    Inputs:
        - **pred**: the prediction tensor, where the `batch_axis` dimension
          ranges over batch size and `axis` dimension ranges over the number
          of classes.
        - **label**: the truth tensor. When `sparse_label` is True, `label`'s
          shape should be `pred`'s shape with the `axis` dimension removed.
          i.e. for `pred` with shape (1,2,3,4) and `axis = 2`, `label`'s shape
          should be (1,2,4) and values should be integers between 0 and 2. If
          `sparse_label` is False, `label`'s shape must be the same as `pred`
          and values should be floats in the range `[0, 1]`.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as label. For example, if label has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).
    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """
    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(WeightedBCEDICE, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits
        self._pool = nn.AvgPool2D(pool_size=(11, 11), strides = (1, 1), padding=0)

    def weighted_dice_coeff(self, F, pred, label, weight):
        smooth = 1.
        w = weight * weight
        pred_y = F.argmax(pred, axis = self._axis) # Returns the indices of the maximum values along an axis.
        intersection = pred_y * label

        # dice coeficient
        # mean
        score = (2 * F.sum(w * intersection, axis=self._batch_axis, exclude=True) + smooth) \
            / (F.sum(w * label, axis=self._batch_axis, exclude=True) + F.sum(w * pred_y, axis=self._batch_axis, exclude=True) + smooth)
        
        return score

    def dice_coeff(self, F, pred, label):
        smooth = 1.
        pred_y = F.argmax(pred, axis = self._axis) # Returns the indices of the maximum values along an axis.
        intersection = pred_y * label

        # dice coeficient
        score = (2 * F.sum(intersection, axis=self._batch_axis, exclude=True) + smooth) \
            / (F.sum(label, axis=self._batch_axis, exclude=True) + F.sum(pred_y, axis=self._batch_axis, exclude=True) + smooth)
        
        return score
        # return 1 - score
        # return - F.log(score)

    def dice_loss(self, F, pred, label):
        loss = 1 - self.dice_coeff(F, pred, label)
        return loss
        # return - F.log(score)

    def weighted_dice_loss(self, F, pred, label, weight):
        loss = 1 - self.weighted_dice_coeff(F, pred, label, weight)
        return loss
        # return - F.log(score)

        
    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.log_softmax(pred, self._axis)
        if self._sparse_label:
            loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(pred*label, axis=self._axis, keepdims=True)

        # This does really well to provide a dynamic weight ballancing for class imbalance but it does require more testing
        # Initial result show better results than standard Weighted BCE loss
        
        # Input data should be 4D in (batch, channel, y, x) but it comes in as (batch,  y, x)
        # Expand shape into (B x C X H x W )
        if True:
            data = mx.ndarray.expand_dims(label, axis=1)
            averaged_mask = self._pool(data)
            weight = F.ones_like(averaged_mask)
            w0 = F.sum(weight)
            weight = 5. * F.exp(-5. * F.abs(averaged_mask - 0.5))
            w1 = F.sum(weight)
            zz = (w0 / w1)
            weight = zz.asnumpy().flat[0]
            self._weight = zz.asnumpy().flat[0]
        else:
            self._weight = None

        # self._weight = None
        # TODO : Implement Tversky Focal Loss
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        # diceloss = self.dice_loss(F, pred, label)
        diceloss = self.weighted_dice_loss(F, pred, label, self._weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True) + diceloss
        # return F.sum(loss, axis=self._batch_axis, exclude=True) + diceloss