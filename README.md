# U-Net with state of the art encoder and decoder

## Task

We are tasked to provide the segmentation of cell boundaries within mouse brain by training a model on the dataset including fruit fly brain image.

(to do) : insert images of origibal and segmentation mask

## Data

In this model, we trained on fruit fly images and mouse brain images found online and validated the model on the provided mouse brain images and masks to select the model and evaluate the performace of this model based in IOU metric. 

(to do): maybe to add the definition of IOU?
 
### training 

* fruit fly image and segmentation mask pair

### test and validation 
* mouse brain image

## Model

As a baseline model, we adopted U-Net, which is characterized with encoder and decoder architecture, because this model is proven to be work well with biological images. 

In order to improve the performance we incorporated couple of changes below.

(to do) : insert the picture of u-net architecture.

* Modified the encoder to Deep Residual Pyramid Net
* incorporated spatial and channelwise squeeze and excitation block at the end of block of u-net
* optional: shakedrop regularization technique to see if it generalizes well

## Result

## Discussion
