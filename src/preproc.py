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

img_size = 256

def data_transforms():
    general_aug = Compose([
        OneOf([
            Transpose(),
            HorizontalFlip(),
            RandomRotate90()
            ]),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=.2),
        OneOf([
            OpticalDistortion(p=0.2),
            GridDistortion(distort_limit=0.1, p=.1),
            IAAPiecewiseAffine(p=0.2),
            ], p=0.2)
        ], p=1)
    image_specific = Compose([
        OneOf([
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
            ], p=0.3)
        ])
    all_transf_pre = [
            transforms.RandomCrop(round(1.2 * img_size))
            ]

    all_transf_after = [
            transforms.RandomCrop(img_size)
            ]
    normalize = [
            transforms.ToTensor()
            ]

    def get_augment(aug):
        def augment(image):
            return Image.fromarray(aug(image=np.array(image))['image'])
        return [augment]

    train_general_transform = transforms.Compose(all_transf_pre + get_augment(general_aug))
    train_img_transform = transforms.Compose(get_augment(image_specific))
    norm_transform = transforms.Compose(all_transf_after + normalize)
    val_transform = transforms.Compose(normalize)

    return train_general_transform, train_img_transform, norm_transform, val_transform
