from config import Config
import matplotlib.pyplot as plt
import numpy as np
import utils

def display_images(loader, nb_imgs):
    for (img, label), _ in zip(loader, range(nb_imgs)):
        img = np.squeeze(img.detach().cpu().numpy())
        label = np.squeeze(label.detach().cpu().numpy())
        plt.gray()
        fig = plt.subplot(1,2,1)
        fig.imshow(img)
        fig = plt.subplot(1,2,2)
        fig.imshow(label)
        plt.show()

if __name__ == '__main__':
    _, loader = utils.get_data()
    nb_imgs = 15
    display_images(loader, nb_imgs)
