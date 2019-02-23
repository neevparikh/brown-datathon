import numpy as np
from torchvision import transforms
from PIL import Image
import random
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, 
    ElasticTransform, ChannelShuffle,RGBShift, Rotate
)

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, r_m=2):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r_m = r_m
       
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        num_regions = random.randint(1, self.r_m)

        for _ in range(num_regions):
            for attempt in range(100):
                area = img.size()[1] * img.size()[2]
           
                target_area = random.uniform(self.sl, self.sh) * area / num_regions
                aspect_ratio = random.uniform(self.r1, 1/self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.size()[2] and h < img.size()[1]:
                    x1 = random.randint(0, img.size()[1] - h)
                    y1 = random.randint(0, img.size()[2] - w)
                    if img.size()[0] == 3:
                        img[0, x1:x1+h, y1:y1+w] = torch.randn(h, w)
                        img[1, x1:x1+h, y1:y1+w] = torch.randn(h, w)
                        img[2, x1:x1+h, y1:y1+w] = torch.randn(h, w)
                    else:
                        img[0, x1:x1+h, y1:y1+w] = torch.randn(h, w)
                    break

        return img

def data_transforms(cutout_prob, min_erase_area, max_erase_area, min_erase_aspect_ratio, max_erase_regions,
        avg, std):
    all_transf = [
        transforms.RandomCrop(200)
    ]
    aug = Compose([
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
            ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            Blur(blur_limit=2, p=.1),
            ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=.2),
        OneOf([
            OpticalDistortion(p=0.2),
            GridDistortion(distort_limit=0.1, p=.1),
            IAAPiecewiseAffine(p=0.2),
            ], p=0.2),
        OneOf([
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
            ], p=0.3),
        ], p=1)

    def augment(image):
        return Image.fromarray(aug(image=np.array(image))['image'])

    train_transf = [
        transforms.RandomHorizontalFlip(),
        augment,
    ]

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize([avg], [std]),
    ]

    erase = [
        RandomErasing(cutout_prob, min_erase_area, max_erase_area, min_erase_aspect_ratio, max_erase_regions)
    ]

    train_transform = transforms.Compose(train_transf + all_transf + normalize + erase)
    valid_transform = transforms.Compose(all_transf + normalize)

    return train_transform, valid_transform
