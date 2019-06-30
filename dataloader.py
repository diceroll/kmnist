from pathlib import Path

import chainer
import numpy as np
import pandas as pd
from albumentations import (Compose, Cutout, HorizontalFlip, LongestMaxSize,
                            PadIfNeeded, RandomCrop, RandomScale, Resize,
                            Rotate, VerticalFlip)
from chainer.datasets import split_dataset_random
from PIL import Image


def mixup(img, img2, alpha=0.2):
    ratio = np.random.beta(alpha, alpha)
    img = ratio * img + (1 - ratio) * img2

    return img, ratio


class KMNIST(chainer.dataset.DatasetMixin):

    def __init__(self, augmentation=None, index=None,
                 drop_index=None, train=True, pseudo_labeling=False):
        assert index is None or drop_index is None

        if train:
            self.imgs = np.load(
                './data/kmnist-train-imgs.npz')['arr_0']
            self.labels = np.load(
                './data/kmnist-train-labels.npz')['arr_0']
        else:
            self.imgs = np.load(
                './data/kmnist-test-imgs.npz')['arr_0']
            self.labels = None

        if index is not None:
            self.imgs = self.imgs[index]
            if self.labels is not None:
                self.labels = self.labels[index]
        elif drop_index is not None:
            self.imgs = np.delete(self.imgs, drop_index, 0)
            if self.labels is not None:
                self.labels = np.delete(self.labels, drop_index, 0)
        self.n_train = len(self.imgs)

        if train and pseudo_labeling:
            df = pd.read_csv('./data/pseudo_labeling.csv')
            add_index = df[df.Prob > 0.9].index
            add_imgs = np.load(
                './data/kmnist-test-imgs.npz')['arr_0'][add_index]
            add_labels = np.array(df[df.Prob > 0.9].Label)
            self.imgs = np.concatenate((self.imgs, add_imgs))
            self.labels = np.append(self.labels, add_labels)

        self.mixup = 0
        self.augmentation = []
        if augmentation is not None:
            processes = []
            for (process, params) in augmentation:
                if process == 'Mixup':
                    self.augmentation.append(Compose(processes, p=1.0))
                    self.mixup = params['p']
                    processes = []
                else:
                    processes.append(eval(process)(**params))
            if len(processes) > 0:
                self.augmentation.append(Compose(processes, p=1.0))

    def __len__(self):
        return len(self.imgs)

    def get_example(self, i):
        img = self.imgs[i].copy().astype(np.float32)

        if len(self.augmentation) > 0:
            img = self.augmentation[0](image=img)['image']

        if self.labels is not None:
            label = np.zeros(4, np.float32)
            label[0] = self.labels[i]

            if np.random.rand() < self.mixup:
                index = np.random.randint(len(self.imgs))
                img2 = self.imgs[index].copy().astype(np.float32)
                if len(self.augmentation) > 0:
                    img2 = self.augmentation[0](image=img2)['image']
                img, ratio = mixup(img, img2, alpha=0.2)
                label[1] = self.labels[index]
                label[2] = ratio
                label[3] = 1 - ratio
                if index >= self.n_train:
                    label[3] *= 0.9
            else:
                label[1] = self.labels[i]
                label[2] = 1
                label[3] = 0
            if i >= self.n_train:
                label[2] *= 0.9

        if len(self.augmentation) > 1:
            img = self.augmentation[1](image=img)['image']

        img = img / 127.5 - 1
        img = img[np.newaxis, :, :]

        if self.labels is None:
            return img

        return img, label


if __name__ == '__main__':

    augmentation = [
        ('Rotate', {'p': 0.8, 'limit': 5}),
        ('PadIfNeeded', {'p': 0.5, 'min_height': 28, 'min_width': 30}),
        ('PadIfNeeded', {'p': 0.5, 'min_height': 30, 'min_width': 28}),
        ('Resize', {'p': 1.0, 'height': 28, 'width': 28}),
        ('RandomScale', {'p': 1.0, 'scale_limit': 0.1}),
        ('PadIfNeeded', {'p': 1.0, 'min_height': 32, 'min_width': 32}),
        ('RandomCrop', {'p': 1.0, 'height': 28, 'width': 28}),
        ('Mixup', {'p': 0.5}),
        ('Cutout', {'p': 0.5, 'num_holes': 4, 'max_h_size': 4,
                    'max_w_size': 4}),
    ]

    sl = slice(0, None, 5)
    train_data = KMNIST(augmentation=augmentation, drop_index=sl)
    print(len(train_data))
    test_data = KMNIST(index=sl)
    print(len(test_data))

    img, label = train_data.get_example(0)
    print(img.shape, label)
