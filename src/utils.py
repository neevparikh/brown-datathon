from torch.utils.data import Dataset
import numpy as np
import os
import preproc
from PIL import Image
from collections import namedtuple
import logging
import torch
import math
import shutil
import dill

data_path = os.path.join('..', '..', 'data')
train_root_dir = os.path.join(data_path, 'train')
test_root_dir = os.path.join(data_path, 'test')
save_path = os.path.join(data_path, 'image_stats.npz')
indexes_file = os.path.join(data_path, 'indexes.npy')
num_folds = 5

image_stats = np.load(save_path)
avg = float(image_stats['avg'])
std = float(image_stats['std'])

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class SceneDataset(Dataset):
    def __init__(self, root_dir, index, transform=None):
        paths = list(map(lambda f : os.path.join(root_dir, f), os.listdir(root_dir)))
        self.items = index(sum([[(os.path.join(p, i_p), i) for i_p in os.listdir(p)] 
            for i, p in enumerate(paths)], [])) 
        self.shape = (256, 256)
        self.num_classes = len(paths)
        self.transform = transform
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, label = self.items[i]
        img = Image.open(p)
        if self.transform:
            img = self.transform(img)
        return (img, torch.tensor(int(label)))

def get_image_stats_scene(seed):
    np.random.seed(seed)
    scene_data = SceneDataset(train_root_dir, lambda x : x)
    #images are grey scale
    var = 0
    avg = 0
    min_dim_x = 1e10
    min_dim_y = 1e10
    for i in range(len(scene_data)):
        img = np.array(scene_data[i][0]).astype(np.float32) / 255.
        d_x, d_y = img.shape
        if d_x < min_dim_x:
            min_dim_x = d_x
        if d_y < min_dim_y:
            min_dim_y = d_y
        var += np.var(img) / len(scene_data)
        avg += np.average(img) / len(scene_data)
    std = np.sqrt(var)
    print("std:", std, "avg:", avg, "min_dim:", (min_dim_x, min_dim_y))
    np.savez(save_path, std=std, avg=avg)
    indexes = np.arange(len(scene_data))
    np.random.shuffle(indexes)
    np.save(indexes_file, indexes)

def get_data(data_path, fold, cutout_prob, min_erase_area, max_erase_area, min_erase_aspect_ratio,
        max_erase_regions):
    indexes = np.load(indexes_file)
    assert fold >= 0 and fold < num_folds
    fold_start = round(indexes.shape[0] * fold / num_folds)
    fold_middle = round(indexes.shape[0] * (fold + 1) / num_folds)
    trn_indexes = np.concatenate((indexes[:fold_start], indexes[fold_middle:]))
    val_indexes = indexes[fold_start:fold_middle]
    indexer = lambda indexes : lambda x : np.array(x)[indexes]
    trn_transform, val_transform = preproc.data_transforms(cutout_prob, min_erase_area, 
            max_erase_area, min_erase_aspect_ratio, max_erase_regions, avg, std)
    trn_data = SceneDataset(root_dir=train_root_dir, index=indexer(trn_indexes), transform=trn_transform)
    val_data = SceneDataset(root_dir=train_root_dir, index=indexer(val_indexes), transform=val_transform)

    num_classes = trn_data.num_classes

    #shape is HW or HWC
    shape = trn_data.shape
    input_channels = 3 if len(shape) == 3 else 1
    assert shape[0] == shape[1], "not expected shape = {}".format(shape)
    input_size = shape[0]

    return [{'input_size': input_size, 'input_channels': input_channels, 'num_classes': num_classes}, trn_data,
            val_data]

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('scene')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.

class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_item(item, f_dir, names):
    filename = os.path.join(f_dir, names[0]+'.pth.tar')
    torch.save(item, filename, pickle_module=dill)
    for name in names[1:]:
        other_filename = os.path.join(f_dir, name+'.pth.tar')
        shutil.copyfile(filename, other_filename)

class ExpFinderSchedule(namedtuple('ExpFinderSchedule', ('lr_start', 'lr_end', 'nb_iters_train'))):
    def __call__(self, t):
        r = t / self.nb_iters_train
        return self.lr_start * (self.lr_end / self.lr_start) ** r

class PiecewiseLinearOrCos(namedtuple('PiecewiseLinear', ('knots', 'vals', 'is_cos'))):
    def __call__(self, t):
        if t <= self.knots[0]:
            return float(self.vals[0])
        if t >= self.knots[-1]:
            return float(self.vals[-1])
        for k in range(len(self.knots)):
            if t < self.knots[k]:
                break

        if self.is_cos[k-1]:
            a = self.vals[k-1]
            b = self.vals[k]
            c = self.knots[k-1]
            d = self.knots[k]

            assert abs(c-d) > 1e-200
            
            return math.cos((t-c)*math.pi/(c-d))*(a-b)/2.+(b-a)/2+a
        else:    
            return np.interp([t], self.knots, self.vals)[0]

if __name__ == '__main__':
    get_image_stats_scene(7)
