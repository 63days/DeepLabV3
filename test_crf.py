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
from torchvision import transforms
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    d = dcrf.DenseCRF2D(512, 512, 2)

    down_method = args.down_method
    up_method = args.up_method
    separable = args.separable

    ds = DataSetWrapper(1, args.num_workers, 0.2)
    test_dl = ds.get_data_loaders(train=False)

    model = Unet(input_dim=1, separable=True,
                 down_method='conv', up_method='transpose')
    model = nn.DataParallel(model).to(device)

    #load_state = torch.load(f'./checkpoint/{down_method}_{up_method}_{separable}.ckpt')
    load_state = torch.load(f'./checkpoint/conv_transpose_True.ckpt')

    model.load_state_dict(load_state['model_state_dict'])
    train_losses = load_state['train_losses']
    val_losses = load_state['val_losses']
    map_list = []
    model.eval()
    with torch.no_grad():
        for (img, label) in test_dl:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            img, label, pred = img.squeeze(0), label.squeeze(0), pred.squeeze(0)
            pred2 = torch.zeros(2, 512, 512).to(device)
            index = torch.cat([torch.zeros(1,512, 512), torch.ones(1,512, 512)], dim=0).long().to(device)
            pred2.scatter_(0, index, torch.cat([1-pred, pred], dim=0))

            U = pred2.cpu().numpy().transpose(2,0,1).reshape((2,-1)) # [2, 512*512]
            U = np.where(a )
            U = -np.log(U)
            d.setUnaryEnergy(U)
            print(U)
            im = np.array(img.cpu(), dtype=np.uint8).transpose(1, 2, 0)
            pairwise_energy = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01,), img=im, chdim=2)
            d.addPairwiseEnergy(pairwise_energy, compat=10)

            Q = d.inference(5)
            map = np.argmax(Q, axis=0).reshape((512, 512))
            proba = np.array(Q)
            print(proba)

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            plt.subplots_adjust(top=1, bottom=0, hspace=0.01)
            ax[0].imshow(img.cpu().numpy().transpose(1,2,0), cmap='gray')
            ax[1].imshow(label.cpu().numpy().transpose(1,2,0), cmap='gray')
            ax[2].imshow(map, cmap='gray')

            plt.show()

            break




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