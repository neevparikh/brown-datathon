from config import Config
import matplotlib.pyplot as plt
import numpy as np
import utils

def display_images(loader, nb_imgs):
    while(True):
        for img_label in loader:
            img_label = np.squeeze(img_label.detach().cpu().numpy())

            print(img_label[1].max())
            print(img_label[0].max())
            print(img_label[1].min())
            print(img_label[0].min())

            plt.gray()
            fig = plt.subplot(1,2,1)
            fig.imshow(img_label[0])
            fig = plt.subplot(1,2,2)
            fig.imshow(img_label[1])
            plt.show()

if __name__ == '__main__':
    _, loader, _= utils.get_data()
    nb_imgs = 15
    display_images(loader, nb_imgs)
