import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BoneDataset(Dataset):

    def __init__(self, list_dir='./dataset', img_dir='./dataset/images', transform=None):
        self.list_dir = list_dir
        self.img_dir = img_dir
        self.transform = transform
        self.img_list = np.genfromtxt(os.path.join(list_dir, 'filelist.txt'), dtype=str)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]

        img = Image.open(self.img_dir+'/original/'+img_name).convert('L')
        label = Image.open(self.img_dir+'/label/'+img_name).convert('L')

        sample = (img, label)
        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        label = label.long().float()
        sample = (img, label)
        #label_onehot = torch.zeros(2, label.size(1), label.size(2)).scatter_(0, label, 1)
        #sample = (img, label_onehot)

        return sample

    def __len__(self):
        return self.img_list.shape[0]

class DataSetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_ratio, test_ratio):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

    def get_data_loaders(self):
        data_augment = self._bone_transform()

        dataset = BoneDataset(transform=self._bone_transform())

        train_dl, valid_dl, test_dl = self.get_train_valid_test_loaders(dataset)

        return train_dl, valid_dl, test_dl

    def _bone_transform(self):
        data_transforms = transforms.Compose([
            SquarePad(),
            transforms.Resize((572, 572)),
            transforms.ToTensor()
        ])
        return data_transforms

    def get_train_valid_test_loaders(self, dataset):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split_1 = int(np.floor(self.valid_ratio * dataset_size))
        split_2 = int(np.floor(self.test_ratio * dataset_size))
        valid_indices, test_indices = indices[:split_1], indices[split_1: split_1+split_2]
        train_indices = indices[split_1+split_2:]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_dl = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler,
                              num_workers=self.num_workers, shuffle=False, pin_memory=True)

        valid_dl = DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler,
                              num_workers=self.num_workers, shuffle=False, pin_memory=True)

        test_dl = DataLoader(dataset, batch_size=self.batch_size, sampler=test_sampler,
                              num_workers=self.num_workers, shuffle=False, pin_memory=True)

        return train_dl, valid_dl, test_dl

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


if __name__ == '__main__':
    print(device)
    dataset = DataSetWrapper(batch_size=1, num_workers=1, valid_ratio=0.2, test_ratio=0.2)
    train_dl, valid_dl, test_dl = dataset.get_data_loaders()

    x, y = next(iter(train_dl))
    print(y.size())
    fig, ax = plt.subplots(1,2)
    y=y.squeeze(0)
    y=torch.argmax(y, 0)
    ax[0].imshow(x.squeeze(), cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(y, cmap='gray')
    ax[1].axis('off')
    plt.show()
