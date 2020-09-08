import torch
from dataloader import BoneDataset, SquarePad
import os
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np


def main():
    img_list = np.genfromtxt('./dataset/image_list.txt', dtype=str)


    for img_name in img_list:
        img = Image.open('./dataset/images/original/'+img_name).convert('L')
        label = Image.open('./dataset/images/label/'+img_name).convert('L')
        sample = (img, label)

        img_name = img_name[:-4]
        ### original image tensor data ###
        sample = pad_resize(sample, 512)
        ori_sample = to_tensor(sample)
        torch.save(ori_sample, f'./dataset/tensor/{img_name}.ts')
        ##################################

        ### augment image tensor data ###
        for i in range(9):
            aug_sample = random_rotate(sample)
            aug_sample = random_resized_crop(aug_sample)
            aug_sample = to_tensor(aug_sample)
            torch.save(aug_sample, f'./dataset/tensor/{img_name}a{i}.ts')
        ##################################

def split():
    total_list = np.genfromtxt('./dataset/data_list.txt', dtype=str)
    np.random.shuffle(total_list)


def pad_resize(sample, S=512):
    img, label = sample
    img, label = SquarePad()(img), SquarePad()(label)
    img, label = transforms.Resize((S, S))(img), transforms.Resize((S, S))(label)

    return img, label

def random_rotate(sample):
    degrees = np.arange(3, 46, 3)
    d = np.random.choice(degrees)
    if random.random() > 0.5:
        d = -d

    img, label = sample
    img = TF.rotate(img, d, resample=Image.BILINEAR)
    label = TF.rotate(label, d, resample=Image.BILINEAR)

    return img, label

def random_resized_crop(sample):
    img, label = sample
    S = img.size[0]

    ratio = np.random.randint(6, 11) * 0.1
    S_prime = int(S * ratio)

    if S_prime == S:
        return sample

    candidate = np.arange(S-S_prime)

    h, w = np.random.choice(candidate, 2)

    img = TF.resized_crop(img, h, w, S_prime, S_prime, S)
    label = TF.resized_crop(label, h, w, S_prime, S_prime, S)

    return img, label

def to_tensor(sample):
    img, label = sample
    img, label = TF.to_tensor(img), TF.to_tensor(label)

    label = (label > 0.8).float()

    return img, label

if __name__ == '__main__':
    total_list = np.genfromtxt('./dataset/data_list.txt', dtype=str)
    np.random.shuffle(total_list)
    indices = list(range(len(total_list)))
    split = int(np.floor(0.2 * len(total_list)))
    train_idx, test_idx = indices[split:], indices[:split]

    train_list = total_list[train_idx]
    test_list = total_list[test_idx]


    np.savetxt('./dataset/train_list.txt', train_list, fmt='%s')
    np.savetxt('./dataset/test_list.txt', test_list, fmt='%s')