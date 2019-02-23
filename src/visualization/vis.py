"""
This is a visualization file to view the dataset images
Uncomment the first two lines to use it on MacOS. 
"""
import matplotlib
matplotlib.use('TkAgg') 
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

# Load mouse 2 data for evaluation.
mice2_data = load('../../data/train-mouse-2012.npz')
mice2_volume = mice2_data['volume']
mice2_label = mice2_data["label"]



blacklist_indices = {
    "fv": [14, 74],
    "mv": [],
    "fl": [14, 74],
    "ml": [],
    "mv2": [],
    "ml2": []
    }


images = {
    "ml": mice_label,
    "mv": mice_volume,
    "fl": fly_label,
    "fv": fly_volume,
    "mv2": mice2_volume,
    "ml2": mice2_label
        }

for i in range(images["mv2"].shape[0]):
    if i not in blacklist_indices["mv2"]:
        plt.imshow(images["mv2"][i]) 
        plt.gray()
        plt.show()

