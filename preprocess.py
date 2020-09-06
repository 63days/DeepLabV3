import torch
import torch.nn as nn
from dataloader import BoneDataset, SquarePad
import os
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np

def main():
    img_list = np.genfromtxt('./dataset/filelist.txt', dtype=str)


    for img_name in img_list:
        img = Image.open('./dataset/images/original/'+img_name).convert('L')
        label = Image.open('./dataset/images/label/'+img_name).convert('L')

        augmentation_save((img, label)) #[10, C, H, W]







def augmentation_save(sample, img_name): #sample: (img, label)
    img, label = sample
    img_name = img_name[:-4]

    img_t = TF.to_tensor(img)
    label_t = TF.to_tensor(label)
    torch.save(img_t,'./dataset/tensor/original/'+img_name+'.ts')
    torch.save(label_t,'./dataset/tensor/label/'+img_name+'.ts')

    img_list = []
    label_list = []
    img_list.append(img_t)
    label_list.append(label_t)

    pad_resize = transforms.Compose([
        SquarePad(),
        transforms.Resize((512, 512))
    ])
    img = pad_resize(img)
    label = pad_resize(label)

    degrees = np.arange(3, 46, 3)
    degree = np.random.choice(degrees, 9)
    mask = 2 * np.random.randint(0, 2, 9) - 1
    degree *= mask

    for i, d in enumerate(degree):
        rot_img = TF.rotate(img, d, resample=Image.BILINEAR)
        rot_label = TF.rotate(label, d, resample=Image.BILINEAR)
        aug_img_name = img_name + f'_{str(i)}'
        rot_img, rot_label = random_resized_crop((rot_img, rot_label))

        rot_img_t = TF.to_tensor(rot_img)
        rot_label_t = TF.to_tensor(rot_label)

        torch.save(rot_img_t, './dataset/tensor/original/' + aug_img_name + '.ts')
        torch.save(rot_label_t, './dataset/tensor/label/' + aug_img_name + '.ts')





def random_resized_crop(sample):
    img, label = sample
    S = img.size[0]

    ratio = np.random.randint(6, 11) * 0.1
    S_prime = int(S * ratio)

    if S_prime == S:
        return sample

    candidate = np.arange(S-S_prime)

    h, w = np.random.choice(candidate, 2)

    img = TF.resizedcrop(img, h, w, S_prime, S_prime, S)
    label = TF.resizedcrop(label, h, w, S_prime, S_prime, S)

    sample = (img, label)

    return sample

if __name__ == '__main__':
    main()