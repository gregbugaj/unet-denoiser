from mxnet.gluon.loss import Loss, _reshape_like
import numpy as np

class ContrastiveLoss(Loss):
    def __init__(self, axis=-1, weight=None, batch_axis=0, **kwargs):
        super(ContrastiveLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self.margin = 6.

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        print('Log info ********* ')
        reshaped = pred
        pred = label

        print(pred.shape)
        print(label.shape)
        print(reshaped.shape)

        # pred  > (1, 2, 64, 256) Batch, Classes, height, width
        # label > (1, 64, 256)    Batch, height, width

        distances = pred - label
        distances_squared = F.sum(F.square(distances), 1, keepdims=True)
        euclidean_distances = F.sqrt(distances_squared + 0.0001)
        d = F.clip(self.margin - euclidean_distances, 0, self.margin)
        loss = (1 - label) * distances_squared + label * F.square(d)
        loss = 0.5*loss

        print('loss = %s' %(loss))
        return loss
