# unet-denoiser


# Feature loss function
Need to implement custom loss function

* pixel loss
* gram matrix loss

https://github.com/mrgloom/awesome-semantic-segmentation
https://medium.com/@junma11/loss-functions-for-medical-image-segmentation-a-taxonomy-cefa5292eec0
https://github.com/karolzak/keras-unet/blob/2b1c6d5ea1846b97ff7fa0f38ec371f20db73020/keras_unet/losses.py


https://thomasdelteil.github.io/NeuralStyleTransfer_MXNet/
https://mxnet.apache.org/versions/1.7/api/python/docs/api/gluon/loss/index.html#mxnet.gluon.loss.SoftmaxCrossEntropyLoss
https://towardsdatascience.com/u-net-deep-learning-colourisation-of-greyscale-images-ee6c1c61aabe
https://towardsdatascience.com/loss-functions-based-on-feature-activation-and-style-loss-2f0b72fd32a9
https://towardsdatascience.com/u-net-b229b32b4a71#:~:text=Loss%20calculation%20in%20UNet&text=UNet%20uses%20a%20rather%20novel,the%20border%20of%20segmented%20objects.&text=First%20of%20all%20pixel%2Dwise,by%20cross%2Dentropy%20loss%20function.

 ## Tensorboard

 ```sh
 tensorboard --logdir ./logs
 ``` 

## Dependencies 

* MXNET >= 1.7 


## Data set
https://primaresearch.org/datasets

https://arxiv.org/pdf/1906.05229.pdf
The usage of U-Net for pre-processing document images (sibgrapi_pi_cv.pdf)

https://github.com/mkisantal/backboned-unet/blob/master/backboned_unet
https://github.com/chinakook/U-Net/blob/master/model_unet.py

## Network Architecture
https://www.programmersought.com/article/92255025083/

## Super-Resolution 
https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2018/ocr_Ankit_Lat_ICPR_2018.pdf

Binary Document Image Super Resolution for Improved Readability and OCR Performance
https://arxiv.org/pdf/1812.02475.pdf

## Loss
https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
