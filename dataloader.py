import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import pickle
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BoneDataset(Dataset):

    def __init__(self, list_dir='./dataset', data_dir='./dataset/tensor', train=True):
        self.list_dir = list_dir
        self.data_dir = data_dir
        self.train = train
        if train is True:
            self.img_list = np.genfromtxt(os.path.join(list_dir, 'train_list.txt'), dtype=str)
        else:
            self.img_list = np.genfromtxt(os.path.join(list_dir, 'test_list.txt'), dtype=str)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        sample = torch.load(os.path.join(self.data_dir, img_name))
        return sample

    def __len__(self):
        return self.img_list.shape[0]

class DataSetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_ratio):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_ratio = valid_ratio

    def get_data_loaders(self, train):
        if train is True:
            dataset = BoneDataset(train=train)

            train_dl, valid_dl = self.get_train_valid_loaders(dataset)

            return train_dl, valid_dl
        else:
            dataset = BoneDataset(train=train)
            test_dl = self.get_test_loader(dataset)
            return test_dl

    def get_train_valid_loaders(self, dataset):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_ratio * dataset_size))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_dl = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler,
                              num_workers=self.num_workers, shuffle=False, pin_memory=True)

        valid_dl = DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler,
                              num_workers=self.num_workers, shuffle=False, pin_memory=True)

        return train_dl, valid_dl

    def get_test_loader(self, dataset):
        test_dl = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                             shuffle=False, pin_memory=True)

class SquarePad(object):
    def __call__(self, img):
        w, h = img.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return transforms.Pad(padding, 0, 'constant')(img)

class Threshold(object):
    def __init__(self, threshold=0.5, value=0):
        self.threshold = threshold
        self.value = value

    def __call__(self, img):
        return nn.Threshold(self.threshold, self.value)(img)

