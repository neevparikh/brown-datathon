from config import Config
import matplotlib.pyplot as plt
import numpy as np
import utils

def display_images(loader, nb_imgs):
    for (img, _), _ in zip(loader, range(nb_imgs)):
        img = np.squeeze(img.detach().cpu().numpy())
        plt.imshow(img)
        plt.gray()
        plt.show()

if __name__ == '__main__':
    config = Config()
    _, loader, _ = utils.get_data(config.data_path, config.fold, config.cutout_prob, config.min_erase_area, 
            config.max_erase_area, config.min_erase_aspect_ratio, config.max_erase_regions)
    nb_imgs = 15
    display_images(loader, nb_imgs)
