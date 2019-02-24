from config import Config
import matplotlib.pyplot as plt
import numpy as np
import utils

def display_images(loader, nb_imgs):
    for img_label, _ in zip(loader, range(nb_imgs)):
        img_label = np.squeeze(img_label.detach().cpu().numpy())
        plt.gray()
        fig = plt.subplot(1,2,1)
        fig.imshow(img_label[0])
        fig = plt.subplot(1,2,2)
        fig.imshow(img_label[1])
        plt.show()

if __name__ == '__main__':
    _, _ , loader= utils.get_data()
    nb_imgs = 15
    display_images(loader, nb_imgs)
