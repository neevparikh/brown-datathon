"""
This is a visualization file to view the dataset images
Uncomment the first two lines to use it on MacOS. 
"""
#import matplotlib
#matplotlib.use('TkAgg') 
import numpy as np
from numpy import load
from matplotlib import pyplot as plt

# Load fruit fly data for training.
fly_data = load('../../data/fruit_fly_volumes.npz')
fly_volume = fly_data['volume']
fly_label = fly_data["label"]

# Load mouse data for evaluation.
mice_data = load('../../data/mouse_volumes.npz')
mice_volume = mice_data['volume']
mice_label = mice_data["label"]
mice_gt = mice_data["gt"]

blacklist_indices = {
        "fv": [14, 74],
    "mv": [],
    "fl": [14, 74],
    "ml": []
    }


images = {
    "ml": mice_label,
    "mgt": mice_gt,
    "mv": mice_volume,
    "fl": fly_label,
    "fv": fly_volume
        }

for i in range(images["mv"].shape[0]):
    if i not in blacklist_indices["mv"]:
        plt.imshow(images["mv"][i]) 
        plt.gray()
        plt.show()

