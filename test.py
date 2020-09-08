import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from model import Unet
from tqdm import tqdm
from dataloader import DataSetWrapper
from utils import Padding, get_IOU
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    down_method = args.down_method
    up_method = args.up_method
    separable = args.separable

    ds = DataSetWrapper(args.batch_size, args.num_workers, 0.2)
    test_dl = ds.get_data_loaders(train=False)

    model = Unet(input_dim=1, separable=separable,
                 down_method='maxpool', up_method='transpose')
    model = nn.DataParallel(model).to(device)

    #load_state = torch.load(f'./checkpoint/{down_method}_{up_method}_{separable}.ckpt')
    load_state = torch.load(f'./checkpoint/maxpool_transpose_False.ckpt')

    model.load_state_dict(load_state['model_state_dict'])
    train_losses = load_state['train_losses']
    val_losses = load_state['val_losses']

    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_dl)
        mIOU = []
        img_list = []
        label_list = []
        pred_list = []
        for (img, label) in pbar:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            pred = Padding()(pred, label.size(3))
            pred = (pred > 0.5).long()
            mIOU.append(get_IOU(pred, label, threshold=0.5))

            img_list.append(img[0].cpu().numpy())
            label_list.append(label[0].cpu().numpy())
            pred_list.append(pred[0].cpu().numpy())

        mIOU = sum(mIOU) / len(mIOU)
        print(f'mIOU: {mIOU*100:.2f}%')

    length = len(img_list)
    print(length)

    fig, ax = plt.subplots(4, 3, figsize=(6, 10))
    plt.subplots_adjust(top=1, bottom=0, hspace=0.01)
    for i in range(4):
        ax[i][0].imshow(img_list[i].transpose((1,2,0)), cmap='gray')
        ax[i][0].axis('off')
        ax[i][1].imshow(label_list[i].transpose((1,2,0)), cmap='gray')
        ax[i][1].axis('off')
        ax[i][2].imshow(pred_list[i].transpose((1,2,0)), cmap='gray')
        ax[i][2].axis('off')

    plt.show()

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