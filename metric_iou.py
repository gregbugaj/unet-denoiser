import numpy as np

def iou_metric(a, b, epsilon=1e-5, conversion_mode='cast', conversion_params={'p1' : .5, 'p2' : .5}):
    """Intersection Over Union Metric

    Args:
        a:                  (numpy array) component a
        b:                  (numpy array) component b
        epsilon:            (float) Small value to prevent division by zereo
        conversion_mode:    (cast|predicate) Array conversion mode to bool
        conversion_params:  (dictionary) Conversion parameter dictionary 

    Returns:
        (float) The Intersect of Union score.
    """
    if conversion_mode == 'cast':
        d1 = a.astype('bool')
        d2 = b.astype('bool')
    elif conversion_mode == 'predicate': 
        d1 = np.where(a > float(conversion_params['p1']), True, False)
        d2 = np.where(b > float(conversion_params['p2']), True, False)        
    else:
        raise ValueError("Unknown conversion type : %s" % (conversion_mode))        

    overlap = d1 * d2 # logical AND
    union = d1 + d2 # logical OR
    iou = overlap.sum() / (union.sum() + epsilon)
    return iou


    
def iou_np(y_true, y_pred, smooth=1.):
    intersection = y_true * y_pred
    union = y_true + y_pred
    return (np.sum(intersection) + smooth) / (np.sum(union - intersection) + smooth)



def iou_thresholded_np(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred_pos = (y_pred > threshold) * 1.0
    intersection = y_true * y_pred_pos
    union = y_true + y_pred_pos
    return (np.sum(intersection) + smooth) / (np.sum(union - intersection) + smooth)