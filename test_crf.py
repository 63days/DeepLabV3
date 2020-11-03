import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from model import Unet
from tqdm import tqdm
from dataloader import DataSetWrapper
from utils import Padding, get_IOU
import numpy as np
import pydensecrf.densecrf as dcrf
from PIL import Image
from torchvision import transforms
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    d = dcrf.DenseCRF2D(512, 512, 2)

    down_method = args.down_method
    up_method = args.up_method
    separable = args.separable

    ds = DataSetWrapper(args.batch_size, args.num_workers, 0.2)
    test_dl = ds.get_data_loaders(train=False)

    model = Unet(input_dim=1, separable=True,
                 down_method='conv', up_method='transpose')
    model = nn.DataParallel(model).to(device)

    load_state = torch.load(f'./checkpoint/conv_transpose_True.ckpt')

    model.load_state_dict(load_state['model_state_dict'])

    model.eval()
    name=0
    with torch.no_grad():
        for (img, label) in test_dl:
            imgs, labels = img.to(device), label.to(device)
            preds = model(img)
            name += 1
            for i in range(args.batch_size):
                img, label, pred = imgs[i,:],labels[i,:],preds[i,:]

                probs = torch.stack([1-pred, pred], dim=0).cpu().numpy()
                img, label  = img.cpu().numpy(), label.cpu().numpy()
                pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.01,), img=img, chdim=0)
                U = unary_from_softmax(probs)
                d = dcrf.DenseCRF2D(512, 512, 2)
                d.setUnaryEnergy(U)
                d.addPairwiseEnergy(pairwise_energy, compat=10)

                Q = d.inference(100)
                map = np.argmax(Q, axis=0).reshape((512, 512))
                print(map.shape)

                img = (255. / img.max() * (img - img.min())).astype(np.uint8)
                label = (255. / label.max() * (label - label.min())).astype(np.uint8)
                pred = (255. / map.max() * (map - map.min())).astype(np.uint8)

                img = Image.fromarray(img[0,:], mode='L')
                label = Image.fromarray(label[0,:], mode='L')
                pred = Image.fromarray(pred, mode='L')

                img.save(f'./results/{name}_{i}_i.png')
                label.save(f'./results/{name}_{i}_l.png')
                pred.save(f'./results/{name}_{i}_p.png')





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the segmentation')

    parser.add_argument(
        '--num_workers',
        type=int,
        default=8
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8
    )
    parser.add_argument(
        '--separable',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--up_method',
        type=str,
        default='bilinear',
        choices=['bilinear', 'transpose']
    )
    parser.add_argument(
        '--down_method',
        type=str,
        default='maxpool',
        choices=['maxpool', 'conv']
    )
    args = parser.parse_args()
    main(args)