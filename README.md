# U-Net with state of the art encoder and decoder

## Task

We are tasked to provide the segmentation of cell boundaries within mouse brain by training a model on the dataset including fruit fly brain image.

(to do) : insert images of origibal and segmentation mask

## Data

In this model, we trained on fruit fly images and mouse brain images found online and validated the model on the provided mouse brain images and masks to select the model and evaluate the performace of this model based in IOU metric, which is defined as 

\begin{equation}
true positive / (true positive + false positive + false negative)
\end{equation}


 
### training 

* fruit fly image and segmentation mask pair
#### data augmentation

In order to generalize better, we applied data augmentation techniques, which are proven to be effective on biomedical image data.

* random elastic deformation
* random shear (rotation, shifting and scaling)
* random contrast and brightness change
* flip

Then, in order not to have blank space as a result of these transformation, we radomly cropped the images to 256 x 256.

### test and validation 
* mouse brain image

## Model

As a baseline model, we adopted U-Net, which is characterized with encoder and decoder architecture, because this model is proven to be work well with biological images. 

<img src = 'https://www.google.com/url?sa=i&source=images&cd=&ved=2ahUKEwiRn8_3z9TgAhUHJt8KHRNWA9EQjRx6BAgBEAU&url=https%3A%2F%2Fchatbotslife.com%2Fsmall-u-net-for-vehicle-detection-9eec216f9fd6&psig=AOvVaw0jyiZpwfc84_UmY9V5qKkV&ust=1551106435677334'>


In order to improve the performance we incorporated couple of changes below.



* Modified the encoder to Deep Residual Pyramid Net
* incorporated spatial and channelwise squeeze and excitation block at the end of block of u-net
* optional: shakedrop regularization technique to see if it generalizes well

## Result

## Discussion

While we have good result for training set, we did not see that the output on test images are good.
This is presumably because the distribution of the dataset is so different.

Actually, we observed that when we included the test images in the training set, the performance improved dramatically.

We tried to address this issue by incorporating further regularization techniques, suchg as shakedrop architecture.

However, we did not observe significant improvement from there. 


## Challenges and future direction
The biggest challenge is the poor generalization.  

